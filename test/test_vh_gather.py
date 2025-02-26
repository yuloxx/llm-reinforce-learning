
def test_vh_gather_env_function():
    from src.sb3vh import vh_env
    from virtualhome.simulation.unity_simulator.comm_unity import UnityCommunication
    from src.sb3vh.env_graph_enum import ActionEnum
    # test virtual home gather food environment
    YOUR_FILE_NAME = "D:\\programs\\windows_exec.v2.2.4\\VirtualHome.exe"
    comm = UnityCommunication(file_name=YOUR_FILE_NAME)
    vh_env = vh_env.VirtualHomeGatherFoodEnv(comm=comm)
    vh_env.reset(options={
        'environment_index': 0,
    })

    def put_food_in_fridge(food_index: int):
        vh_env.step(action=(ActionEnum.WALK_TO_FOOD, food_index, 0))  # '<char0> [walk] <salmon> (328)'
        vh_env.step(action=(ActionEnum.GRAB, food_index, 0))  # '<char0> [grab] <salmon> (328)'
        vh_env.step(action=(ActionEnum.WALK_TO_OBJECT, 0, 6))  # '<char0> [walk] <fridge> (314)
        vh_env.step(action=(ActionEnum.OPEN, 0, 6))  # '<char0> [walk] <fridge> (306)'
        vh_env.step(action=(ActionEnum.PUTIN, food_index, 6))  # '<char0> [putin] <salmon> (328) <fridge> (306)'
        vh_env.step(action=(ActionEnum.CLOSE, 0, 6))  # '<char0> [close] <fridge> (306)'

    food_index_list = [0, 1, 2, 4, 6, 12, 18]
    for food_index in food_index_list:
        if food_index == 18:
            print('aaa')
        put_food_in_fridge(food_index)
    vh_env.step(action=(ActionEnum.STOP, 0, 0))

    # vh_env.step(action=(ActionEnum.GRAB, 0, 0))
    a = [f'\'{inst}\'' for inst in vh_env.vh_metadata['instruction_list']]
    print(f'[{',\n    '.join(a)}]')

def test_vh_gather_env_checker():
    from stable_baselines3.common.env_checker import check_env
    from src.sb3vh import vh_env
    from virtualhome.simulation.unity_simulator.comm_unity import UnityCommunication
    YOUR_FILE_NAME = "D:\\programs\\windows_exec.v2.2.4\\VirtualHome.exe"
    comm = UnityCommunication(file_name=YOUR_FILE_NAME)
    vh_env = vh_env.VirtualHomeGatherFoodEnv(comm=comm)
    vh_env.reset(options={
        'environment_index': 0,
    })
    check_env(vh_env, warn=True, skip_render_check=True)

def test_vh_render_script():
    from virtualhome.simulation.unity_simulator.comm_unity import UnityCommunication

    s = ['<char0> [walk] <salmon> (328)',
    '<char0> [grab] <salmon> (328)',
    '<char0> [walk] <fridge> (306)',
    '<char0> [open] <fridge> (306)',
    '<char0> [putin] <salmon> (328) <fridge> (306)',
    '<char0> [close] <fridge> (306)',
    '<char0> [walk] <apple> (439)',
    '<char0> [grab] <apple> (439)',
    '<char0> [walk] <fridge> (306)',
    '<char0> [open] <fridge> (306)',
    '<char0> [putin] <apple> (439) <fridge> (306)',
    '<char0> [close] <fridge> (306)',
    '<char0> [walk] <bananas> (440)',
    '<char0> [grab] <bananas> (440)',
    '<char0> [walk] <fridge> (306)',
    '<char0> [open] <fridge> (306)',
    '<char0> [putin] <bananas> (440) <fridge> (306)',
    '<char0> [close] <fridge> (306)',
    '<char0> [walk] <peach> (443)',
    '<char0> [grab] <peach> (443)',
    '<char0> [walk] <fridge> (306)',
    '<char0> [open] <fridge> (306)',
    '<char0> [putin] <peach> (443) <fridge> (306)',
    '<char0> [close] <fridge> (306)',
    '<char0> [walk] <pie> (320)',
    '<char0> [grab] <pie> (320)',
    '<char0> [walk] <fridge> (306)',
    '<char0> [open] <fridge> (306)',
    '<char0> [putin] <pie> (320) <fridge> (306)',
    '<char0> [close] <fridge> (306)',
    '<char0> [walk] <plum> (445)',
    '<char0> [grab] <plum> (445)',
    '<char0> [walk] <fridge> (306)',
    '<char0> [open] <fridge> (306)',
    '<char0> [putin] <plum> (445) <fridge> (306)',
    '<char0> [close] <fridge> (306)',
    '<char0> [walk] <chocolatesyrup> (332)',
    '<char0> [grab] <chocolatesyrup> (332)',
    '<char0> [walk] <fridge> (306)',
    '<char0> [open] <fridge> (306)',
    '<char0> [putin] <chocolatesyrup> (332) <fridge> (306)',
    '<char0> [close] <fridge> (306)']

    YOUR_FILE_NAME = "D:\\programs\\windows_exec.v2.2.4\\VirtualHome.exe"
    comm = UnityCommunication(file_name=YOUR_FILE_NAME)
    res = comm.reset(0)
    print(res)
    comm.add_character('Chars/Male1')
    res = comm.render_script(s, recording=True, skip_animation=False)
    print(res)


if __name__ == '__main__':
    test_vh_gather_env_function()
    # test_vh_gather_env_checker()
    # test_vh_render_script()
