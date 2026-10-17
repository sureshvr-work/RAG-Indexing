"""Shared AND filters for the explicitly synthetic lab metadata fields."""

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.core.models import Document


class MetadataFilter(BaseModel):
    model_config = ConfigDict(extra='forbid')
    groups: list[str] = Field(default_factory=list, max_length=20)
    review_statuses: list[str] = Field(default_factory=list, max_length=20)
    year_min: int | None = Field(default=None, ge=0, le=9999)
    year_max: int | None = Field(default=None, ge=0, le=9999)

    @model_validator(mode='after')
    def valid_range(self):
        if self.year_min is not None and self.year_max is not None and self.year_min > self.year_max:
            raise ValueError('minimum year must not exceed maximum year')
        if any(not value or len(value) > 100 for value in self.groups + self.review_statuses):
            raise ValueError('filter values must be nonempty strings of at most 100 characters')
        return self

    def active(self) -> bool:
        return bool(self.groups or self.review_statuses or self.year_min is not None or self.year_max is not None)

    def matches(self, document: Document) -> bool:
        metadata = document.metadata
        if self.groups and metadata.get('lab_group') not in self.groups:
            return False
        if self.review_statuses and metadata.get('lab_review_status') not in self.review_statuses:
            return False
        if self.year_min is not None or self.year_max is not None:
            year = metadata.get('lab_year')
            if type(year) is not int:
                return False
            if self.year_min is not None and year < self.year_min:
                return False
            if self.year_max is not None and year > self.year_max:
                return False
        return True
