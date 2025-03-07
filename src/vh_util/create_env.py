from virtualhome.simulation.unity_simulator.comm_unity import UnityCommunication

def get_basic_environment_graph(
        virtualhome_exec_path: str,
        environment_index: int = 0,
):
    comm = UnityCommunication(file_name=virtualhome_exec_path)
    try:
        res = comm.reset(environment_index)
        if not res:
            raise ValueError("Error: reset virtual home failed")
        res = comm.add_character()
        if not res:
            raise ValueError("Error: add character failed")
        res, g = comm.environment_graph()
        if not res:
            raise ValueError("Error: get environment graph failed")
    finally:
        comm.close()
    return g

