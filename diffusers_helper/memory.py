# By lllyasviel


import torch
import psutil
import logging

logger = logging.getLogger(__name__)

# Detect available devices
cpu = torch.device('cpu')
if torch.backends.mps.is_available():
    gpu = torch.device('mps')
else:
    gpu = cpu  # Fallback to CPU if MPS not available
    logger.warning("MPS not available, falling back to CPU")

# Track loaded models
gpu_complete_modules = []

def get_available_memory_gb():
    """Get available memory for MPS or system RAM for CPU fallback"""
    try:
        if torch.backends.mps.is_available():
            total_memory = torch.mps.recommended_max_memory()
            used_memory = torch.mps.driver_allocated_memory()
            available_memory = total_memory - used_memory
        else:
            available_memory = psutil.virtual_memory().available
        return available_memory / (1024 ** 3)
    except Exception as e:
        logger.warning(f"Error getting memory info: {e}")
        return psutil.virtual_memory().available / (1024 ** 3)

def empty_cache():
    """Clear MPS memory cache if available"""
    if torch.backends.mps.is_available():
        try:
            torch.mps.empty_cache()
        except Exception as e:
            logger.warning(f"Error clearing MPS cache: {e}")

def move_model_to_device(model, target_device):
    """Move model to specified device and clear cache"""
    logger.info(f'Moving {model.__class__.__name__} to {target_device}')
    try:
        model.to(device=target_device)
        empty_cache()
    except Exception as e:
        logger.error(f"Error moving model to {target_device}: {e}")
        raise

def offload_model_to_cpu(model):
    """Offload model to CPU to free up GPU memory"""
    logger.info(f'Offloading {model.__class__.__name__} to CPU')
    try:
        model.to(device=cpu)
        empty_cache()
    except Exception as e:
        logger.error(f"Error offloading model to CPU: {e}")
        raise

def unload_complete_models(*args):
    """Unload models from GPU memory"""
    for m in gpu_complete_modules + list(args):
        offload_model_to_cpu(m)
        logger.info(f'Unloaded {m.__class__.__name__} as complete')
    
    gpu_complete_modules.clear()
    empty_cache()

def load_model_as_complete(model, target_device, unload=True):
    """Load model to device, optionally unloading other models first"""
    if unload:
        unload_complete_models()
    
    move_model_to_device(model, target_device)
    logger.info(f'Loaded {model.__class__.__name__} to {target_device} as complete')
    gpu_complete_modules.append(model)

# Keep DynamicSwapInstaller for compatibility, but simplify it
class DynamicSwapInstaller:
    """Simplified model device management for MPS"""
    @staticmethod
    def install_model(model: torch.nn.Module, **kwargs):
        move_model_to_device(model, kwargs.get('device', gpu))
    
    @staticmethod
    def uninstall_model(model: torch.nn.Module):
        offload_model_to_cpu(model)


def fake_diffusers_current_device(model: torch.nn.Module, target_device: torch.device):
    """Ensure model's scale_shift_table is on the correct device"""
    if hasattr(model, 'scale_shift_table'):
        model.scale_shift_table.data = model.scale_shift_table.data.to(target_device)
        return

    for k, p in model.named_modules():
        if hasattr(p, 'weight'):
            p.to(target_device)
            return


def move_model_to_device_with_memory_preservation(model, target_device, preserved_memory_gb=0):
    print(f'Moving {model.__class__.__name__} to {target_device} with preserved memory: {preserved_memory_gb} GB')

    for m in model.modules():
        if get_available_memory_gb() <= preserved_memory_gb:
            empty_cache()
            return

        if hasattr(m, 'weight'):
            m.to(device=target_device)

    model.to(device=target_device)
    empty_cache()
    return


def offload_model_from_device_for_memory_preservation(model, target_device, preserved_memory_gb=0):
    print(f'Offloading {model.__class__.__name__} from {target_device} to preserve memory: {preserved_memory_gb} GB')

    if target_device.type == 'cuda':
        for m in model.modules():
            if get_available_memory_gb() >= preserved_memory_gb:
                empty_cache()
                return

            if hasattr(m, 'weight'):
                m.to(device=cpu)
    else:
        # For MPS, just move the model directly
        model.to(device=cpu)

    model.to(device=cpu)
    empty_cache()
    return
