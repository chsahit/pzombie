import numpy as np
from pydrake.all import (
    BasicVector,
    JacobianWrtVariable,
    LeafSystem,
    MultibodyForces,
    Value,
    ValueProducer,
)


class CartesianStiffnessController(LeafSystem):
    def __init__(self, plant):
        LeafSystem.__init__(self)
        self.plant = plant

        self.panda = self.plant.GetModelInstanceByName("panda")
        num_states = self.plant.num_multibody_states(self.panda)
        self.num_q = self.plant.num_positions(self.panda)
        self.num_q_all = self.plant.num_positions()

        self.panda_body_frame = self.plant.GetBodyByName("panda_hand").body_frame()
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
            "desired_state",
            BasicVector(36 + 9 + 9),  # stiffness R^{6x6} + hand_q + hand_qdot
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

    def _compute_damping_mat(self, K_cart, K_q, J):
        K_cart_is_diagonal = np.array_equal(np.diag(np.diag(K_cart)), K_cart)
        if K_cart_is_diagonal:
            B_cart = 3 * np.sqrt(K_cart)
            B = J.T @ B_cart @ J
            B = np.hstack((B, np.zeros((7, 2))))
            B = np.vstack((B, np.zeros((2, 9))))
            return B
        else:
            B = np.eye(9)
            for i in range(9):
                B[i, i] = 3 * np.sqrt(K_q[i, i])
            return B

    def _circle_difference(self, qd, q0):
        q_err = np.zeros((9,))
        for i in range(7):
            diff = qd[i] - q0[i]
            q_err[i] = (diff + np.pi) % (2 * np.pi) - np.pi
        q_err[7] = qd[7] - q0[7]
        q_err[8] = qd[8] - q0[8]
        return q_err

    def CalcOutputForce(self, context, output):
        plant_context = self.get_cache_entry(self.plant_context_cache_index_).Eval(
            context
        )
        applied_forces = self.get_cache_entry(self.applied_forces_cache_index_).Eval(
            context
        )
        J_g = self.get_cache_entry(self.jacobian_cache_idx_).Eval(context)

        tau = self.plant.CalcInverseDynamics(
            plant_context, np.zeros((self.plant.num_velocities(),)), applied_forces
        )
        Cv = self.plant.CalcBiasTerm(plant_context)
        tau -= Cv
        tau = self.plant.GetVelocitiesFromArray(self.panda, tau)

        x = self.get_input_port_estimated_state().Eval(context)
        x_d_K = self.get_input_port_desired_state().Eval(context)

        Kp = np.reshape(x_d_K[:36], (6, 6))
        x_d = x_d_K[36:]

        kp_q = J_g.T @ Kp @ J_g
        finger_gains = np.array(
            [[0, 0, 0, 0, 0, 0, 0, 1000, 0], [0, 0, 0, 0, 0, 0, 0, 0, 1000]]
        )
        kp_q = np.hstack((kp_q, np.zeros((7, 2))))
        kp_q = np.vstack((kp_q, finger_gains))
        kd = self._compute_damping_mat(Kp, kp_q, J_g)
        q_err = self._circle_difference(x_d[: self.num_q], x[: self.num_q])
        qd_err = x_d[-self.num_q :] - x[-self.num_q :]
        spring_damper_F = kp_q @ q_err + kd @ qd_err
        spring_damper_F_mag = np.linalg.norm(spring_damper_F[:7])
        if spring_damper_F_mag > 20:
            spring_damper_F[:7] = (spring_damper_F[:7] / spring_damper_F_mag) * 20
        tau += spring_damper_F
        output.SetFromVector(tau)
