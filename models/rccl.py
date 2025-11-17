from pydantic import BaseModel, Field, model_validator, ConfigDict, field_validator
from typing import Annotated, Literal 
import math

NonNegativeInt = Annotated[int, Field(ge=0)]
PositiveInt = Annotated[int, Field(gt=0)]
NonNegativeFloat = Annotated[float, Field(ge=0.0)]
Collective = Literal['AllReduce', 'AllGather', 'Scatter', 'Gather', 'ReduceScatter', 'SendRecv', 'AllToAll', 'AllToAllV', 'Broadcast']
Type = Literal[
    'int8', 'int32', 'int64', 
    'uint8', 'uint32', 'uint64',
    'float', 'double', 
    'half', 'bfloat16',
    'fp8_e4m3', 'fp8_e5m2'
]
Redop = Literal['sum', 'prod', 'min', 'max', 'avg', 'all', 'none']
InPlace = Literal[0, 1]

class RcclTestsSingleNodeRaw(BaseModel):
    """
    This class represents the schema for single noderccl-test results, while serializing rccl-test input
    if we don't adhere to this schema, we fail immediately preventing weird behaviour later on
    in the processing pipeline
    """
    model_config = ConfigDict(frozen=True)
    numCycle: NonNegativeInt
    name: Collective
    size: PositiveInt
    type: Type
    redop: Redop
    inPlace: InPlace
    time: NonNegativeFloat
    algBw: NonNegativeFloat
    busBw: NonNegativeFloat
    wrong: int

    @field_validator('time', 'algBw', 'busBw')
    @classmethod
    def validate_not_nan_inf(cls, v: float, info) -> float:
        """Ensure no NaN/Inf values in measurements."""
        if math.isnan(v) or math.isinf(v):
            raise ValueError(f'{info.field_name} cannot be NaN/Inf, got {v}')
        return v


class RcclTestsMultinodeRaw(BaseModel):
    """
    This class represents the schema for multi node rccl-test results, while serializing rccl-test input
    if we don't adhere to this schema, we fail immediately preventing weird behaviour later on
    in the processing pipeline
    """
    model_config = ConfigDict(frozen=True)
    numCycle: NonNegativeInt
    name: Collective
    nodes: PositiveInt
    ranks: PositiveInt
    ranksPerNode: PositiveInt
    gpusPerRank: PositiveInt
    size: PositiveInt
    type: Type
    redop: Redop
    inPlace: InPlace
    time: NonNegativeFloat
    algBw: NonNegativeFloat
    busBw: NonNegativeFloat
    wrong: int

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

    @field_validator('time', 'algBw', 'busBw')
    @classmethod
    def validate_not_nan_inf(cls, v: float, info) -> float:
        """Ensure no NaN/Inf values in measurements."""
        if math.isnan(v) or math.isinf(v):
            raise ValueError(f'{info.field_name} cannot be NaN/Inf, got {v}')
        return v

class RcclTestsAggregated(BaseModel):
    """
    This class represents the aggregated schema for rccl-test results
    """
    # Grouping keys
    model_config = ConfigDict(frozen=True, populate_by_name=True)
    name: Collective = Field(alias='collective')
    size: PositiveInt 
    type: Type 

    # Aggregated metrics
    busBw_mean: NonNegativeFloat = Field(alias='busbw_mean')
    busBw_std: NonNegativeFloat = Field(alias='busbw_std')
    algBw_mean: NonNegativeFloat = Field(alias='algbw_mean')
    algBw_std: NonNegativeFloat = Field(alias='algbw_std')
    time_mean: NonNegativeFloat = Field(alias='time_mean')
    time_std: NonNegativeFloat = Field(alias='time_std')

    #Metadata
    num_runs: NonNegativeInt = Field(alias='numCycles', description='Number of cycles aggregated')

    @field_validator('busBw_std', 'algBw_std', 'time_std')
    @classmethod
    def handle_nan_std(cls, v: float, info) -> float:
        """
        Convert NaN (from single-value std) to 0.0.
        Pandas returns NaN for std of single value, which is correct mathematically,
        but we interpret it as 0 variability.
        """
        if math.isnan(v):
            return 0.0
        if math.isinf(v):
            raise ValueError(f'{info.field_name} cannot be Inf')
        return v