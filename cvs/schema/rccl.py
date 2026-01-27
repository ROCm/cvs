# std libs
from typing import Annotated, Literal, Optional
import math

# pypdantic libs
from pydantic import BaseModel, Field, model_validator, ConfigDict, field_validator

NonNegativeInt = Annotated[int, Field(ge=0)]
PositiveInt = Annotated[int, Field(gt=0)]
NonNegativeFloat = Annotated[float, Field(ge=0.0)]
Collective = Literal[
    'AllReduce', 'AllGather', 'Scatter', 'Gather', 'ReduceScatter', 'SendRecv', 'AllToAll', 'AllToAllV', 'Broadcast'
]
Type = Literal[
    'int8', 'int32', 'int64', 'uint8', 'uint32', 'uint64', 'float', 'double', 'half', 'bfloat16', 'fp8_e4m3', 'fp8_e5m2'
]
Redop = Literal['sum', 'prod', 'min', 'max', 'avg', 'all', 'none']
InPlace = Literal[0, 1]


class RcclTests(BaseModel):
    model_config = ConfigDict(frozen=True)
    numCycle: NonNegativeInt
    name: Collective
    size: NonNegativeInt
    type: Type
    redop: Redop
    inPlace: InPlace

    @field_validator('name', mode='before')
    @classmethod
    def normalize_collective_name(cls, v: str) -> str:
        """Normalize collective names to match expected format."""
        # Handle case variations from RCCL test output
        normalization_map = {
            'allreduce': 'AllReduce',
            'allgather': 'AllGather',
            'scatter': 'Scatter',
            'gather': 'Gather',
            'reducescatter': 'ReduceScatter',
            'sendrecv': 'SendRecv',
            'alltoall': 'AllToAll',  # Maps "AlltoAll" -> "AllToAll"
            'alltoallv': 'AllToAllV',
            'broadcast': 'Broadcast',
        }

        # Try exact match first
        if v in [
            'AllReduce',
            'AllGather',
            'Scatter',
            'Gather',
            'ReduceScatter',
            'SendRecv',
            'AllToAll',
            'AllToAllV',
            'Broadcast',
        ]:
            return v

        # Try case-insensitive normalization
        v_lower = v.lower().replace('_', '').replace('-', '')
        if v_lower in normalization_map:
            return normalization_map[v_lower]

        # If no match, return original (will fail validation with clear error)
        return v

    time: NonNegativeFloat
    algBw: NonNegativeFloat
    busBw: NonNegativeFloat
    wrong: int

    @field_validator('wrong', mode='before')
    @classmethod
    def normalize_wrong_field(cls, v) -> int:
        """Handle 'N/A' or string values in wrong field."""
        # If it's already an int, return it
        if isinstance(v, int):
            return v

        # Treat NaN as "no errors" (0 wrong)
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            return 0

        # Handle string values
        if isinstance(v, str):
            v = v.strip()
            # 'N/A' means no errors (0 wrong)
            if v.upper() in ['N/A', 'NA', 'NONE', '']:
                return 0
            # Try to parse as int
            try:
                return int(v)
            except ValueError:
                # If can't parse, assume no errors
                return 0

        # For any other type, try to convert to int
        try:
            return int(v)
        except (ValueError, TypeError):
            return 0

    @model_validator(mode='after')
    def validate_wrong_is_zero(self):
        """
        Enforce correctness of rccl-tests iteration results.
        After normalization, any positive `wrong` indicates corrupted/invalid results.
        """
        if self.wrong < 0:
            raise ValueError(f'wrong must be >= 0, got {self.wrong}')
        if self.wrong > 0:
            raise ValueError(
                f"SEVERE DATA CORRUPTION: rccl-tests reported non-zero '#wrong' after normalization "
                f"(wrong={self.wrong}). Results are invalid/corrupted."
            )
        return self

    @field_validator('time', 'algBw', 'busBw')
    @classmethod
    def validate_not_nan_inf(cls, v: float, info) -> float:
        """Ensure no NaN/Inf values in measurements."""
        if math.isnan(v) or math.isinf(v):
            raise ValueError(f'{info.field_name} cannot be NaN/Inf, got {v}')
        return v


class RcclTestsMultinodeRaw(RcclTests):
    """
    This class represents the schema for multi node rccl-test results, while serializing rccl-test input
    if we don't adhere to this schema, we fail immediately preventing weird behaviour later on
    in the processing pipeline
    """

    nodes: PositiveInt
    ranks: PositiveInt
    ranksPerNode: PositiveInt
    gpusPerRank: PositiveInt

    @model_validator(mode='after')
    def validate_ranks_relationship(self):
        """Ensure ranks = nodes * ranksPerNode."""
        expected_ranks = self.nodes * self.ranksPerNode
        if self.ranks != expected_ranks:
            raise ValueError(
                f"ranks ({self.ranks}) must equal nodes ({self.nodes}) × "
                f"ranksPerNode ({self.ranksPerNode}) = {expected_ranks}"
            )
        return self


class RcclTestsAggregated(BaseModel):
    """
    This class represents the aggregated schema for rccl-test results
    """

    # Grouping keys
    model_config = ConfigDict(frozen=True, populate_by_name=True)
    name: Collective = Field(alias='collective')
    size: NonNegativeInt
    type: Type
    inPlace: InPlace

    # Metadata
    num_runs: PositiveInt = Field(description='Number of cycles aggregated')

    # Aggregated metrics
    busBw_mean: NonNegativeFloat
    busBw_std: NonNegativeFloat
    algBw_mean: NonNegativeFloat
    algBw_std: NonNegativeFloat
    time_mean: NonNegativeFloat
    time_std: NonNegativeFloat

    # Multinode metadata (optional, None for single-node tests)
    nodes: Optional[PositiveInt] = None
    ranks: Optional[PositiveInt] = None
    ranksPerNode: Optional[PositiveInt] = None
    gpusPerRank: Optional[PositiveInt] = None

    @field_validator('busBw_mean', 'algBw_mean', 'time_mean', mode='before')
    @classmethod
    def handle_nan_mean(cls, v, info) -> float:
        """
        Validate mean values - should never be NaN or Inf.
        Using mode='before' to provide better error messages.
        """
        # Handle None or missing values
        if v is None:
            raise ValueError(f'{info.field_name} cannot be None')

        # Convert to float if needed
        try:
            v_float = float(v)
        except (ValueError, TypeError):
            raise ValueError(f'{info.field_name} must be a valid number')

        # Check for NaN and Inf
        if math.isnan(v_float):
            raise ValueError(f'{info.field_name} cannot be NaN')
        if math.isinf(v_float):
            raise ValueError(f'{info.field_name} cannot be Inf')
        if v_float < 0:
            raise ValueError(f'{info.field_name} must be >= 0')
        return v_float

    @field_validator('busBw_std', 'algBw_std', 'time_std', mode='before')
    @classmethod
    def handle_nan_std(cls, v, info) -> float:
        """
        Convert NaN (from single-value std) to 0.0.
        Pandas returns NaN for std of single value, which is correct mathematically,
        but we interpret it as 0 variability.

        Using mode='before' to run before NonNegativeFloat constraint check.
        """
        # Handle None or missing values
        if v is None:
            return 0.0

        # Convert to float if needed
        try:
            v_float = float(v)
        except (ValueError, TypeError):
            return 0.0

        # Check for NaN and Inf
        if math.isnan(v_float):
            return 0.0
        if math.isinf(v_float):
            raise ValueError(f'{info.field_name} cannot be Inf')
        return v_float
