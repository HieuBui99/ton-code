import enum


class ToolKind(enum.Enum):
    READ_ONLY = "read_only"
    FILE_EDIT = "file_edit"
    OTHER = "other"
