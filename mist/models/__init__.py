"""Initialize and register all MIST model variants."""

# Trigger registration of all model variants.
from mist.models.mednext.mednext_registry import (  # noqa: F401
    create_mednext_base,
    create_mednext_large,
    create_mednext_medium,
    create_mednext_small,
)
from mist.models.mgnets.mgnets_registry import (  # noqa: F401
    create_fmgnet,
    create_wnet,
)
from mist.models.nnunet.nnunet_registry import create_nnunet, create_nnunet_pocket  # noqa: F401
from mist.models.resenc.resenc_registry import create_resenc  # noqa: F401
from mist.models.swinunetr.swinunetr_registry import (  # noqa: F401
    create_swinunetr_base,
    create_swinunetr_large,
    create_swinunetr_small,
)
