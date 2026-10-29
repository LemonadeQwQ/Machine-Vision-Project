"""Train-fitted age/sex/site features. Diagnostic and source fields are never inputs."""
import math
from dataclasses import dataclass
from pathlib import Path

try:
    from .dataset import read_csv
except ImportError:
    from dataset import read_csv

FIELDS = ('age_years', 'sex', 'anatomical_region')


def read_metadata(path, include_skin_tone=False):
    fields = (*FIELDS, "skin_tone_class") if include_skin_tone else FIELDS
    rows = read_csv(path, ['lesion_id', *fields])
    result = {}
    for row in rows:
        key = row['lesion_id']
        if not key or key != key.strip() or key in result:
            raise ValueError(f'Empty, padded or duplicate metadata lesion_id: {key!r}')
        result[key] = {field: row[field].strip() for field in fields}
    return result


def _age(value):
    if value == '':
        return None
    age = float(value)
    if not math.isfinite(age) or not 0 <= age <= 130:
        raise ValueError(f'Invalid age_years: {value!r}')
    return age


def _category(value):
    return value.strip().lower()


def _skin_tone(value):
    value = str(value).strip()
    if not value:
        return None
    number = float(value)
    if not math.isfinite(number) or not number.is_integer() or not 0 <= number <= 5:
        raise ValueError(f'Invalid skin_tone_class: {value!r}; expected integer 0..5 or blank')
    return int(number)


@dataclass(frozen=True)
class MetadataEncoder:
    age_mean: float
    age_std: float
    sex_categories: tuple
    site_categories: tuple
    include_skin_tone: bool = False

    @classmethod
    def fit(cls, rows, include_skin_tone=False):
        rows = list(rows)
        if not rows:
            raise ValueError('Metadata encoder needs training rows')
        ages = [age for row in rows if (age := _age(row['age_years'])) is not None]
        if not ages:
            raise ValueError('No observed training ages')
        mean = sum(ages) / len(ages)
        std = math.sqrt(sum((age - mean) ** 2 for age in ages) / len(ages)) or 1.0
        categories = lambda field: tuple(sorted({_category(r[field]) for r in rows} - {''}))
        if include_skin_tone:
            for row in rows:
                _skin_tone(row['skin_tone_class'])
        return cls(mean, std, categories('sex'), categories('anatomical_region'), include_skin_tone)

    @property
    def feature_names(self):
        return ['age_zscore', 'age_missing', 'sex_unknown',
                *[f'sex_{c}' for c in self.sex_categories], 'site_unknown',
                *[f'site_{c}' for c in self.site_categories]] + (['skin_tone_unknown'] + [f'skin_tone_{i}' for i in range(6)] if self.include_skin_tone else [])

    @property
    def output_dim(self):
        return len(self.feature_names)

    def transform(self, row):
        age = _age(row['age_years'])
        values = [0.0 if age is None else (age - self.age_mean) / self.age_std,
                  float(age is None)]
        for field, categories in (('sex', self.sex_categories),
                                  ('anatomical_region', self.site_categories)):
            category = _category(row[field])
            values.extend([float(category not in categories),
                           *[float(category == known) for known in categories]])
        if self.include_skin_tone:
            tone = _skin_tone(row['skin_tone_class'])
            values.extend([float(tone is None), *[float(tone == i) for i in range(6)]])
        return values

    def to_dict(self):
        return {'version': 2 if self.include_skin_tone else 1, 'fields': list(FIELDS) + (['skin_tone_class'] if self.include_skin_tone else []), 'age_mean': self.age_mean,
                'age_std': self.age_std, 'sex_categories': list(self.sex_categories),
                'site_categories': list(self.site_categories), 'feature_names': self.feature_names,
                'fitted_on': 'training split only',
                'missing': 'age: mean imputation plus indicator; categories: unknown bucket'}

    @classmethod
    def from_dict(cls, config):
        include_skin_tone = config['version'] == 2
        expected = list(FIELDS) + (['skin_tone_class'] if include_skin_tone else [])
        if config['version'] not in (1, 2) or config['fields'] != expected:
            raise ValueError('Unknown metadata schema')
        encoder = cls(float(config['age_mean']), float(config['age_std']),
                      tuple(config['sex_categories']), tuple(config['site_categories']), include_skin_tone)
        if not math.isfinite(encoder.age_mean) or not math.isfinite(encoder.age_std) or encoder.age_std <= 0:
            raise ValueError('Invalid metadata normalization')
        if encoder.feature_names != config['feature_names']:
            raise ValueError('Metadata feature order mismatch')
        return encoder


def fit_training_encoder(split_csv, lesions_csv, include_skin_tone=False):
    metadata = read_metadata(lesions_csv, include_skin_tone=include_skin_tone)
    rows = read_csv(Path(split_csv), ['lesion_id'])
    ids = [row['lesion_id'] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate training lesion_id')
    missing = set(ids) - metadata.keys()
    if missing:
        raise ValueError(f'Missing training metadata: {sorted(missing)[:5]}')
    return MetadataEncoder.fit((metadata[key] for key in ids), include_skin_tone=include_skin_tone)
