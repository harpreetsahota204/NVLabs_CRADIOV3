import logging
import math
import warnings

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.ndimage import gaussian_filter
from skimage.transform import resize
from sklearn.decomposition import PCA
from torchvision.transforms.functional import pil_to_tensor

import fiftyone.core.labels as fol
import fiftyone.core.models as fom
import fiftyone.utils.torch as fout

logger = logging.getLogger(__name__)

class TorchRadioModelConfig(fout.TorchImageModelConfig):
    """Configuration for running a :class:`TorchRadioModel`.

    See :class:`fiftyone.utils.torch.TorchImageModelConfig` for additional
    arguments.

    Args:
        model_version: the RADIO model version to load
        model_path: path to the saved model file on disk
        output_type: what to return - "summary", "spatial", or "both"
        feature_format: "NCHW" or "NLC" for spatial features format
        use_external_preprocessor: whether to use external preprocessing
    """

    def __init__(self, d):
        super().__init__(d)

        self.model_version = self.parse_string(d, "model_version", default="c-radio_v3-h")
        self.model_path = self.parse_string(d, "model_path")
        self.output_type = self.parse_string(d, "output_type", default="summary")
        self.feature_format = self.parse_string(d, "feature_format", default="NCHW")
        self.use_external_preprocessor = self.parse_bool(d, "use_external_preprocessor", default=False)
        self.use_mixed_precision = self.parse_bool(d, "use_mixed_precision", default=True)
        self.apply_smoothing = self.parse_bool(d, "apply_smoothing", default=True)
        self.smoothing_sigma = self.parse_number(d, "smoothing_sigma", default=1.51)

class TorchRadioModel(fout.TorchImageModel):
    """Wrapper for RADIO models from https://github.com/NVlabs/RADIO.

    Args:
        config: a :class:`TorchRadioModelConfig`
    """

    def __init__(self, config):
        super().__init__(config)
        
        # Load the RADIO model and setup preprocessor
        self._radio_model = self._load_radio_model()
        if config.use_external_preprocessor:
            self._conditioner = self._radio_model.make_preprocessor_external()
        else:
            self._conditioner = None
            
    def _check_mixed_precision_support(self):
        """Check if the current GPU supports mixed precision with bfloat16."""
        if not self._using_gpu:
            return False
            
        try:
            # Check GPU capability
            if torch.cuda.is_available():
                device_capability = torch.cuda.get_device_capability(self._device)
                # bfloat16 is supported on Ampere (8.0+) and newer architectures
                return device_capability[0] >= 8
            return False
        except Exception as e:
            logger.warning(f"Could not determine mixed precision support: {e}")
            return False

    def _load_model(self, config):
        """Load the RADIO model from disk."""
        import os
        
        if not os.path.exists(config.model_path):
            raise FileNotFoundError(f"Model file not found: {config.model_path}")
        
        logger.info(f"Loading RADIO model from {config.model_path}")
        
        # Load the saved model data
        checkpoint = torch.load(config.model_path, map_location='cpu')  # Load to CPU first
        model_version = checkpoint['model_version']
        
        # First load the model architecture from torch hub
        model = torch.hub.load(
            'NVlabs/RADIO', 
            'radio_model', 
            version=model_version, 
            progress=False,  # Don't download again
            skip_validation=True
        )
        
        # Load the saved state dict
        model.load_state_dict(checkpoint['model_state_dict'])
        
        return model

    def _load_radio_model(self):
        """Load and setup the RADIO model."""
        model = self._load_model(self.config)
        # Ensure the model is on the correct device and in eval mode
        model = model.to(self._device)
        model.eval()
        return model

    def _preprocess_image(self, img):
        """Preprocess a single image for RADIO model.
        
        Args:
            img: PIL Image, numpy array, or torch tensor
            
        Returns:
            preprocessed tensor ready for RADIO model
        """
        # Convert to PIL if needed
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img)
        elif isinstance(img, torch.Tensor):
            # Convert tensor back to PIL for consistent preprocessing
            if img.dim() == 3:  # CHW
                img = img.permute(1, 2, 0)  # HWC
            img = img.cpu().numpy()
            if img.dtype != np.uint8:
                img = (img * 255).astype(np.uint8)
            img = Image.fromarray(img)
        
        if not isinstance(img, Image.Image):
            raise ValueError(f"Unsupported image type: {type(img)}")
        
        # Ensure RGB
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Convert to tensor and normalize to [0, 1]
        # Make sure to put tensor on the correct device from the start
        x = pil_to_tensor(img).to(dtype=torch.float32)
        x.div_(255.0)  # RADIO expects values between 0 and 1
        x = x.to(self._device)  # Move to device after preprocessing
        
        return x

    def _predict_all(self, imgs):
        """Apply RADIO model to batch of images."""
        
        # Debug: check input types and devices
        logger.debug(f"Input imgs type: {type(imgs)}, length: {len(imgs) if hasattr(imgs, '__len__') else 'unknown'}")
        if len(imgs) > 0:
            logger.debug(f"First img type: {type(imgs[0])}")
            if isinstance(imgs[0], torch.Tensor):
                logger.debug(f"First img device: {imgs[0].device}")
        
        # Preprocess images if needed
        if self._preprocess and self._transforms is not None:
            imgs = [self._transforms(img) for img in imgs]
            # Ensure all tensors are on the correct device
            imgs = [img.to(self._device) if isinstance(img, torch.Tensor) else img for img in imgs]
        else:
            # Apply our custom preprocessing (already handles device placement)
            imgs = [self._preprocess_image(img) for img in imgs]

        # Debug: check preprocessed images
        logger.debug(f"After preprocessing, first img device: {imgs[0].device if isinstance(imgs[0], torch.Tensor) else 'not tensor'}")

        # Process each image individually due to dynamic resizing
        summaries = []
        spatial_features_list = []
        
        for i, img in enumerate(imgs):
            try:
                # Ensure tensor is on the correct device
                if isinstance(img, torch.Tensor):
                    img = img.to(self._device)
                    logger.debug(f"Image {i} device after .to(): {img.device}")
                
                # Add batch dimension if needed
                if img.dim() == 3:
                    img = img.unsqueeze(0)
                
                # Get nearest supported resolution and resize
                nearest_res = self._radio_model.get_nearest_supported_resolution(*img.shape[-2:])
                img_resized = F.interpolate(img, nearest_res, mode='bilinear', align_corners=False)
                
                logger.debug(f"Image {i} resized device: {img_resized.device}")
                
                # Set optimal window size for E-RADIO models
                if "e-radio" in self.config.model_version:
                    self._radio_model.model.set_optimal_window_size(img_resized.shape[2:])
                
                # Apply external conditioning if using external preprocessor
                if self._conditioner is not None:
                    img_resized = self._conditioner(img_resized)
                    logger.debug(f"Image {i} after conditioning device: {img_resized.device}")
                
                # Forward pass with optional mixed precision
                use_mixed_precision = (
                    getattr(self.config, 'use_mixed_precision', False) and 
                    getattr(self, '_mixed_precision_supported', False) and 
                    self._using_gpu
                )
                
                if use_mixed_precision:
                    with torch.autocast('cuda', dtype=torch.bfloat16):
                        summary, spatial = self._radio_model(img_resized, feature_fmt=self.config.feature_format)
                else:
                    with torch.no_grad():
                        summary, spatial = self._radio_model(img_resized, feature_fmt=self.config.feature_format)
                
                logger.debug(f"Image {i} output devices - summary: {summary.device}, spatial: {spatial.device}")
                
                summaries.append(summary)
                spatial_features_list.append(spatial)
                
            except Exception as e:
                logger.error(f"Error processing image {i}: {e}")
                logger.error(f"Image {i} shape: {img.shape if hasattr(img, 'shape') else 'no shape'}")
                logger.error(f"Image {i} device: {img.device if isinstance(img, torch.Tensor) else 'not tensor'}")
                logger.error(f"Model device: {next(self._radio_model.parameters()).device}")
                raise
        
        # Stack results
        try:
            batch_summary = torch.cat(summaries, dim=0)
            batch_spatial = torch.cat(spatial_features_list, dim=0)
        except Exception as e:
            logger.error(f"Error stacking results: {e}")
            logger.error(f"Summary devices: {[s.device for s in summaries]}")
            logger.error(f"Spatial devices: {[s.device for s in spatial_features_list]}")
            raise
        
        # Return based on output type
        if self.config.output_type == "summary":
            output = batch_summary
        elif self.config.output_type == "spatial":
            output = batch_spatial
        else:
            raise ValueError(f"Unknown output_type: {self.config.output_type}")
        
        # Process output if we have an output processor
        if self._output_processor is not None:
            # Collect original frame sizes for batch
            frame_sizes = []
            for img in imgs:
                if hasattr(img, 'shape') and len(img.shape) >= 2:
                    h, w = img.shape[-2:]
                    frame_sizes.append((w, h))  # (width, height) format
                else:
                    # Fallback - this shouldn't happen but just in case
                    frame_sizes.append((224, 224))
            
            return self._output_processor(
                output, frame_sizes, confidence_thresh=self.config.confidence_thresh
            )
        
        # Return raw features as numpy arrays for embeddings
        return [output[i].detach().cpu().numpy() for i in range(len(imgs))]

    def _check_mixed_precision_support(self):
        """Check if the current GPU supports mixed precision with bfloat16."""
        if not self._using_gpu:
            return False
            
        try:
            # Check GPU capability
            if torch.cuda.is_available():
                device_capability = torch.cuda.get_device_capability(self._device)
                # bfloat16 is supported on Ampere (8.0+) and newer architectures
                return device_capability[0] >= 8
            return False
        except Exception as e:
            logger.warning(f"Could not determine mixed precision support: {e}")
            return False


class RadioOutputProcessor(fout.OutputProcessor):
    """Output processor for RADIO models that handles embeddings output."""
    
    def __init__(self, output_type="summary", **kwargs):
        super().__init__(**kwargs)
        self.output_type = output_type
        
    def __call__(self, output, frame_size, confidence_thresh=None):
        """Process RADIO model output into embeddings.
        
        Args:
            output: tensor from RADIO model
            frame_size: (width, height) - not used for embeddings
            confidence_thresh: not used for embeddings
            
        Returns:
            list of numpy arrays containing embeddings
        """
        batch_size = output.shape[0]
        return [output[i].detach().cpu().numpy() for i in range(batch_size)]

class SpatialHeatmapOutputProcessor(fout.OutputProcessor):
    """Improved spatial heatmap processor for RADIO with NCHW features and smoothing."""

    def __init__(self, apply_smoothing=True, smoothing_sigma=1.0, **kwargs):
        super().__init__(**kwargs)
        self.apply_smoothing = apply_smoothing
        self.smoothing_sigma = smoothing_sigma

    def __call__(self, output, frame_sizes, confidence_thresh=None):
        """
        Args:
            output: torch.Tensor of shape [B, C, H, W]
            frame_sizes: list of (width, height) for each image
            confidence_thresh: unused

        Returns:
            List of fol.Heatmap instances
        """
        batch_size = output.shape[0]
        heatmaps = []

        for i in range(batch_size):
            spatial = output[i].detach().cpu().numpy()  # [C, H, W]
            C, H, W = spatial.shape

            # Flatten spatial grid to [H*W, C] for PCA
            reshaped = spatial.reshape(C, -1).T  # [H*W, C]

            try:
                # PCA to reduce channels to 1D attention per pixel
                pca = PCA(n_components=1)
                attention_1d = pca.fit_transform(reshaped).reshape(H, W)
            except Exception as e:
                # Fallback to simple mean over channels
                warnings.warn(f"PCA failed on image {i}: {e}. Falling back to channel mean.")
                attention_1d = spatial.mean(axis=0)  # [H, W]

            # Optional smoothing
            if self.apply_smoothing:
                attention_1d = gaussian_filter(attention_1d, sigma=self.smoothing_sigma)

            # Resize to match original image dimensions
            orig_w, orig_h = frame_sizes[i]
            attention_resized = resize(
                attention_1d,
                (orig_h, orig_w),
                preserve_range=True,
                anti_aliasing=True
            )

            # Normalize to uint8 [0, 255]
            att_min, att_max = attention_resized.min(), attention_resized.max()
            if att_max > att_min:
                attention_uint8 = ((attention_resized - att_min) / (att_max - att_min) * 255).astype(np.uint8)
            else:
                attention_uint8 = np.zeros_like(attention_resized, dtype=np.uint8)

            heatmap = fol.Heatmap(
                map=attention_uint8,
                range=[0, 255]
            )
            heatmaps.append(heatmap)

        return heatmaps
