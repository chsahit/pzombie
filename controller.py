import numpy as np
from pydrake.all import (
    BasicVector,
    LeafSystem,
    JacobianWrtVariable,
    MultibodyForces,
    Value,
    ValueProducer,
)


class JointStiffnessController(LeafSystem):
    def __init__(self, plant):
        LeafSystem.__init__(self)
        self.plant = plant

        self.panda = self.plant.GetModelInstanceByName("panda")
        num_states = self.plant.num_multibody_states(self.panda)
        self.num_q = self.plant.num_positions(self.panda)
        self.num_q_all = self.plant.num_positions()

        self.panda_body_frame = self.plant.GetBodyByName(
            "panda_hand", self.panda
        ).body_frame()
        self.W = self.plant.world_frame()
        self.panda_start_pos = plant.GetJointByName(
            "panda_joint1", self.panda
        ).position_start()
        self.panda_end_pos = plant.GetJointByName(
            "panda_joint7", self.panda
        ).position_start()

        self.input_port_index_estimated_state_ = self.DeclareVectorInputPort(
            "estimated_state", num_states
        ).get_index()
        self.input_port_index_desired_state_ = self.DeclareVectorInputPort(
            "desired_state", BasicVector(num_states)
        ).get_index()
        self.input_port_stiffness = self.DeclareVectorInputPort(
            "stiffness", 6
        ).get_index()
        self.output_port_index_force_ = self.DeclareVectorOutputPort(
            "generalized_force",
            BasicVector(self.num_q),
            self.CalcOutputForce,
            {
                self.all_input_ports_ticket(),
            },
        ).get_index()

        self.plant_context = plant.CreateDefaultContext()
        self.pc_value = Value(self.plant_context)

        self.plant_context_cache_index_ = self.DeclareCacheEntry(
            "plant_context_cache",
            ValueProducer(allocate=self.pc_value.Clone, calc=self.SetMultibodyContext),
            {
                self.input_port_ticket(
                    self.get_input_port_estimated_state().get_index()
                ),
            },
        ).cache_index()

        self.applied_forces_cache_index_ = self.DeclareCacheEntry(
            "applied_forces_cache",
            ValueProducer(
                allocate=Value(MultibodyForces(self.plant)).Clone,
                calc=self.CalcMultibodyForces,
            ),
            {
                self.cache_entry_ticket(self.plant_context_cache_index_),
            },
        ).cache_index()

        self.J = np.zeros((6, 7))
        self.jacobian_cache_idx_ = self.DeclareCacheEntry(
            "jacobian_cache",
            ValueProducer(
                allocate=Value(self.J).Clone,
                calc=self.CalcJacobian,
            ),
            {
                self.cache_entry_ticket(self.plant_context_cache_index_),
            },
        ).cache_index()

    def get_input_port_estimated_state(self):
        return self.get_input_port(self.input_port_index_estimated_state_)

    def get_input_port_desired_state(self):
        return self.get_input_port(self.input_port_index_desired_state_)

    def get_output_port_generalized_force(self):
        return self.get_output_port(self.output_port_index_force_)

    def SetMultibodyContext(self, context, abstract_value):
        state = self.get_input_port_estimated_state().Eval(context)
        self.plant.SetPositionsAndVelocities(
            abstract_value.get_mutable_value(), self.panda, state
        )

    def CalcJacobian(self, context, cache_val):
        plant_context = self.get_cache_entry(self.plant_context_cache_index_).Eval(
            context
        )
        J = self.plant.CalcJacobianSpatialVelocity(
            plant_context,
            JacobianWrtVariable.kQDot,
            self.panda_body_frame,
            np.array([0, 0, 0]),
            self.W,
            self.W,
        )
        J_g = J[:, self.panda_start_pos : self.panda_end_pos + 1]
        np.copyto(cache_val.get_mutable_value(), J_g)

    def CalcMultibodyForces(self, context, cache_val):
        plant_context = self.get_cache_entry(self.plant_context_cache_index_).Eval(
            context
        )
        self.plant.CalcForceElementsContribution(
            plant_context, cache_val.get_mutable_value()
        )


