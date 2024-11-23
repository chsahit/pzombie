import numpy as np
from pydrake.all import (
    AbstractValue,
    BasicVector,
    ContactResults,
    JacobianWrtVariable,
    LeafSystem,
    Value,
    ValueProducer,
)

from pzombie import components


class PolicySystem(LeafSystem):
    # TODO: need output caching
    def __init__(self, plant, asset_indices, scene_graph):
        LeafSystem.__init__(self)
        self.plant = plant
        # K is 36 dimensional, panda has 9 positions and 9 vels
        self.output_port_xd = self.DeclareVectorOutputPort(
            "out", BasicVector(36 + 9 + 9), self.CalcOuput
        )
        self.panda_body_frame = self.plant.GetBodyByName("panda_hand").body_frame()
        self.W = self.plant.world_frame()
        self.policy = None
        self.input_port_index_estimated_state_ = self.DeclareVectorInputPort(
            "estimated_state", plant.num_multibody_states()
        ).get_index()
        self.input_port_index_contact_ = self.DeclareAbstractInputPort(
            "contact_state", AbstractValue.Make(ContactResults())
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
        self.J = np.zeros((6, 7))
        self.panda_start_pos = plant.GetJointByName(
            "panda_joint1", self.plant.GetModelInstanceByName("panda")
        ).position_start()
        self.panda_end_pos = plant.GetJointByName(
            "panda_joint7", self.plant.GetModelInstanceByName("panda")
        ).position_start()

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

        self.panda = plant.GetModelInstanceByName("panda")
        self.np = plant.num_positions()
        self.nv = plant.num_velocities()
        self.asset_indices = asset_indices
        self.inspector = scene_graph.model_inspector()

    def get_input_port_estimated_state(self):
        return self.get_input_port(self.input_port_index_estimated_state_)

    def get_input_port_contact(self):
        return self.get_input_port(self.input_port_index_contact_)

    def SetMultibodyContext(self, context, abstract_value):
        state = self.get_input_port_estimated_state().Eval(context)
        self.plant.SetPositionsAndVelocities(abstract_value.get_mutable_value(), state)

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

    def unpack_contact_results(self, contact_results):
        contacts = list()
        for h_contact in range(contact_results.num_hydroelastic_contacts()):
            surface = contact_results.hydroelastic_contact_info(
                h_contact
            ).contact_surface()
            A = self.plant.GetBodyFromFrameId(self.inspector.GetFrameId(surface.id_M()))
            B = self.plant.GetBodyFromFrameId(self.inspector.GetFrameId(surface.id_N()))
            centroid = surface.centroid()
            F = contact_results.hydroelastic_contact_info(h_contact).F_Ac_W()
            contacts.append(components.Contact(A.name(), B.name(), centroid, F))
        for p_contact_idx in range(contact_results.num_point_pair_contacts()):
            p_contact = contact_results.point_pair_contact_info(p_contact_idx)
            A = self.plant.get_body(p_contact.bodyA_index())
            B = self.plant.get_body(p_contact.bodyB_index())
            p = p_contact.contact_point()
            F = p_contact.contact_force()
            contacts.append(components.Contact(A.name(), B.name(), p, F))

        return contacts

    def CalcOuput(self, context, output):
        assert self.policy is not None
        x_full = self.get_input_port_estimated_state().Eval(context)
        contact_results = self.get_input_port_contact().Eval(context)
        contacts = self.unpack_contact_results(contact_results)
        x_vec = x_full[: self.np]
        v_vec = x_full[self.np : self.np + self.nv]
        x = dict()
        panda_q = self.plant.GetPositionsFromArray(self.panda, x_vec)
        panda_v = self.plant.GetVelocitiesFromArray(self.panda, v_vec)
        x["panda"] = np.concatenate([panda_q, panda_v])
        for asset_name, asset_idx in self.asset_indices.items():
            asset_q = self.plant.GetPositionsFromArray(asset_idx, x_vec)
            asset_v = self.plant.GetVelocitiesFromArray(asset_idx, v_vec)
            x[asset_name] = np.concatenate([asset_q, asset_v])
        t = context.get_time()
        J = self.get_cache_entry(self.jacobian_cache_idx_).Eval(context)
        s = components.State(x, contacts, J)
        a = self.policy(s, t).serialize()
        output.SetFromVector(a)
