from pydrake.all import BasicVector, LeafSystem, Value, ValueProducer


class PolicySystem(LeafSystem):
    # TODO: need output caching
    def __init__(self, plant, num_states: int, asset_indices):
        LeafSystem.__init__(self)
        self.plant = plant
        self.output_port_xd = self.DeclareVectorOutputPort(
            "out", BasicVector(36 + 9 + 9), self.CalcOuput
        )
        self.policy = None
        self.input_port_index_estimated_state_ = self.DeclareVectorInputPort(
            "estimated_state", num_states
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
        self.panda = plant.GetModelInstanceByName("panda")
        self.asset_indices = asset_indices

    def get_input_port_estimated_state(self):
        return self.get_input_port(self.input_port_index_estimated_state_)

    def SetMultibodyContext(self, context, abstract_value):
        state = self.get_input_port_estimated_state().Eval(context)
        self.plant.SetPositionsAndVelocities(abstract_value.get_mutable_value(), state)

    def CalcOuput(self, context, output):
        # raise NotImplementedError("only using panda state")
        assert self.policy is not None
        x_vec = self.get_input_port_estimated_state().Eval(context)[
            : self.plant.num_positions()
        ]
        x = dict()
        x["panda"] = self.plant.GetPositionsFromArray(self.panda, x_vec)
        for asset_name, asset_idx in self.asset_indices.items():
            x[asset_name] = self.plant.GetPositionsFromArray(asset_idx, x_vec)
        t = context.get_time()
        a = self.policy(x, t).serialize()
        output.SetFromVector(a)
