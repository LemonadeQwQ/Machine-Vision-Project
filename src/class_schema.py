"""Versioned class orders; the legacy default remains the original 11 classes."""

CLASSES = (
    "AKIEC", "BCC", "BEN_OTH", "BKL", "DF", "INF",
    "MAL_OTH", "MEL", "NV", "SCCKA", "VASC",
)

NINE_CLASSES = tuple(name for name in CLASSES if name not in ("BEN_OTH", "MAL_OTH"))
# 八类实验只从九类中移除 INF；始终显式保存类别顺序，不能由 CSV 自动推断。
EIGHT_CLASSES = tuple(name for name in NINE_CLASSES if name != "INF")


def validate_class_order(class_names):
    """Require an explicit known order; never infer class indices from a CSV."""
    names = tuple(class_names)
    if names not in (CLASSES, NINE_CLASSES, EIGHT_CLASSES):
        raise ValueError("Expected the fixed 11-class, 9-class or 8-class order")
    return names
