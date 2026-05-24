"""Factor plugin system - here add your custom factors.

BaseFactor protocol:
  - add_to_graph(graph, values, step_idx, sensor_data, context)
  - add_prior(graph, values, sensor_data, context)
  - sensor_fields() -> List[str]
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List
import gtsam


class BaseFactor(ABC):
    """Abstract base for all factor-graph factors.

    Each factor implements two hooks:
      add_to_graph  — called at every keyframe to insert factor(s) into the graph.
      add_prior     — called once for the first frame to add prior constraints.
    """

    @abstractmethod
    def add_to_graph(self,
                     graph: gtsam.NonlinearFactorGraph,
                     values: gtsam.Values,
                     step_idx: int,
                     sensor_data: Dict[str, Any],
                     context: Dict[str, Any]) -> None:
        """Insert factor(s) connecting state at step_idx-1 → step_idx.

        Args:
            graph:       The batch NonlinearFactorGraph being built for this keyframe.
            values:      Initial Values estimate for the keyframe.
            step_idx:    Current keyframe index (0-based).
            sensor_data: Dict with sensor readings from the bridge.
            context:     Dict with keys: 'pose_key', 'vel_key', 'bias_key' callables
                         and 'pim' (PreintegratedCombinedMeasurements object).
        """
        # it is called at every keyframe/timestamp
        ...

    @abstractmethod
    def add_prior(self,
                  graph: gtsam.NonlinearFactorGraph,
                  values: gtsam.Values,
                  sensor_data: Dict[str, Any],
                  context: Dict[str, Any]) -> None:
        """Add prior factors for the first keyframe (step_idx == 0).

        Args:
            graph:       The initial NonlinearFactorGraph.
            values:      Initial Values estimate.
            sensor_data: Dict with sensor readings from the bridge.
            context:     Dict with key callables.
        """
        ...
        # it is only called at the beginning, so add prior here

    @abstractmethod
    def sensor_fields(self) -> List[str]:
        """Return list of sensor_data keys this factor needs."""
        ...
        # ts prevents keyerror, avoids passing unused data, lets the beidge/simulator validaate inputs at start
        # example: fk returns ["joint_pos"], contac treturns ["foot_contact", "imu_acc"]

class FactorRegistry:
    """Ordered collection of factors, dispatched at each keyframe."""

    def __init__(self) -> None:
        self._factors: List[BaseFactor] = []

    def register(self, factor: BaseFactor) -> None:
        """Add a factor to the registry."""
        self._factors.append(factor)
        # simple append. call this during initialisation registry.register(MyImuFactor())

    @property
    def factors(self) -> List[BaseFactor]:
        return list(self._factors)
    # return a copy to not modify the internal list

    def add_all_to_graph(self,
                         graph: gtsam.NonlinearFactorGraph,
                         values: gtsam.Values,
                         step_idx: int,
                         sensor_data: Dict[str, Any],
                         context: Dict[str, Any]) -> None:
        """Dispatch add_to_graph to every registered factor."""
        for factor in self._factors:
            factor.add_to_graph(graph, values, step_idx, sensor_data, context)
    # at each keyframe is called once. iterates through every factor and delefates graph construction

    def add_all_priors(self,
                       graph: gtsam.NonlinearFactorGraph,
                       values: gtsam.Values,
                       sensor_data: Dict[str, Any],
                       context: Dict[str, Any]) -> None:
        """Dispatch add_prior to every registered factor."""
        for factor in self._factors:
            factor.add_prior(graph, values, sensor_data, context)
    # same as previous one, but only for initial step case

    def required_sensor_fields(self) -> List[str]:
        """Union of all sensor_fields() across registered factors."""
        fields: List[str] = []
        for factor in self._factors:
            for f in factor.sensor_fields():
                if f not in fields:
                    fields.append(f)
        return fields
    # collect all unique sensor keys required by the system
    # validates that bdidge actually produces the keys
    # skips copying irrelevant arrays
    # will fail fast, if u forgot to implement a sensor hook