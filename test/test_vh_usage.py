# formatted to ./docs/tutorial/vh_envdata.ipynb

import pprint
from virtualhome.simulation.unity_simulator import comm_unity

# init UnityCommunication
YOUR_FILE_NAME = "D:\\programs\\windows_exec.v2.2.4\\VirtualHome.exe"
CHARACTER_NAME = 'Chars/Male1'

food_list = ['salmon', 'apple', 'bananas', 'pancake', 'peach', 'pear', 'pie', 'potato',
             'salad', 'tomato', 'wine', 'beer', 'plum', 'orange', 'milkshake', 'mincedmeat',
             'lemon', 'juice', 'chocolatesyrup', 'chicken', 'carrot']

room_list = ['kitchen', 'livingroom', 'kitchen', 'bedroom']
object_list = ['microwave', 'coffeetable', 'kitchentable', 'wallshelf', 'kitchencounter', 'desk', 'fridge', 'bookshelf',
               'stove']



def find_food_test():
    from src.vh_util.env_graph import query_node_id_by_classname
    comm = comm_unity.UnityCommunication(file_name=YOUR_FILE_NAME)
    for i in range(6):
        comm.reset(i)
        res, g = comm.environment_graph()
        food_query_list = [f'food name: {food_name}, id: {query_node_id_by_classname(food_name, g)}'
                     for food_name in food_list if query_node_id_by_classname(food_name, g) is not None]
        print(f'---food query in env{i}----')
        print('\n'.join(food_query_list))


def find_object_test():
    from src.vh_util.env_graph import query_node_id_by_classname
    comm = comm_unity.UnityCommunication(file_name=YOUR_FILE_NAME)
    for i in range(6):
        comm.reset(i)
        res, g = comm.environment_graph()
        object_query_list = [
            f'object name: {object_name}, id: {query_node_id_by_classname(object_name, g)}'
            for object_name in object_list if query_node_id_by_classname(object_name, g) is not None
        ]
        print(f'---object query in env{i}----')
        print('\n'.join(object_query_list))


def find_character_relation_test():
    from src.vh_util.env_graph import select_character_relations, query_character_relations, query_character_id
    comm = comm_unity.UnityCommunication(file_name=YOUR_FILE_NAME)
    for i in range(6):
        comm.reset(i)
        comm.add_character(CHARACTER_NAME)
        res, g = comm.environment_graph()
        character_id = query_character_id(g)[0]
        rel = select_character_relations(character_id, g)

        print(f'---character query in env{i}----')
        pprint.pprint(rel)
        print()
        rel_str = query_character_relations(character_id, g)
        print('\n'.join(rel_str))


def operate_food_test():
    from src.vh_util.env_graph import query_node_id_by_classname, query_relations_by_node_id
    from src.vh_util.env_graph import query_character_relations, query_character_id
    comm = comm_unity.UnityCommunication(file_name=YOUR_FILE_NAME)
    comm.reset(0)
    comm.add_character(CHARACTER_NAME)


    res, g = comm.environment_graph()
    character_id = query_character_id(g)[0]

    salmon_id = query_node_id_by_classname('salmon', g)
    salmon_relations = query_relations_by_node_id(salmon_id, g)
    print('salmon relations:')
    pprint.pprint(salmon_relations)
    print()

    res, g = comm.environment_graph()
    character_relation = query_character_relations(character_id, g)
    print('character relation: initial')
    pprint.pprint(character_relation)

    comm.render_script(script=[
        f'<char0> [walk] <salmon> ({salmon_id})'
    ], recording=False, skip_animation=True)

    res, g = comm.environment_graph()
    character_relation = query_character_relations(character_id, g)
    print('character relation: walk to salmon')
    pprint.pprint(character_relation)

    comm.render_script(script=[
        f'<char0> [grab] <salmon> ({salmon_id})'
    ], recording=False, skip_animation=True)

    res, g = comm.environment_graph()
    character_relation = query_character_relations(character_id, g)
    print('character relation: grabbed salmon')
    pprint.pprint(character_relation)

    salmon_relations = query_relations_by_node_id(salmon_id, g)
    print('salmon relations:')
    pprint.pprint(salmon_relations)
    print()


def object_properties_test():
    from src.vh_util.env_graph import select_node_by_classname

    comm = comm_unity.UnityCommunication(file_name=YOUR_FILE_NAME)
    comm.reset(0)

    res, g = comm.environment_graph()

    # mircowave node:
    # 'properties': ['CAN_OPEN', 'HAS_SWITCH', 'CONTAINERS', 'HAS_PLUG']

    microwave_node = select_node_by_classname('microwave', g)
    print(microwave_node['properties'])

    coffee_table_node = select_node_by_classname('coffeetable', g)
    print(coffee_table_node['properties'])


def operate_salmon_script_test():

    comm = comm_unity.UnityCommunication(file_name=YOUR_FILE_NAME)
    res = comm.reset(0)
    res, g = comm.environment_graph()
    comm.add_character('Chars/Male1')

    script = [
        '<char0> [walk] <salmon> (328)',
        '<char0> [grab] <salmon> (328)',
        '<char0> [walk] <fridge> (306)',
        '<char0> [open] <fridge> (306)',
        '<char0> [putin] <salmon> (328) <fridge> (306)',
        '<char0> [close] <fridge> (306)',
    ]
    res = comm.render_script(script, recording=True, skip_animation=False)
    print(res)

def operate_food_script_test():
    from src.vh_util.env_graph import query_node_id_by_classname

    comm = comm_unity.UnityCommunication(file_name=YOUR_FILE_NAME)

    for i in range(7):
        print(f'--- environment {i} ---')
        for food in food_list:
            res = comm.reset(i)
            res, g = comm.environment_graph()
            comm.add_character('Chars/Male1')

            food_id = query_node_id_by_classname(food, g)
            fridge_id = query_node_id_by_classname('fridge', g)

            if food_id is not None:
                script = [
                    f'<char0> [walk] <{food}> ({food_id})',
                    f'<char0> [grab] <{food}> ({food_id})',
                    f'<char0> [walk] <fridge> ({fridge_id})',
                    f'<char0> [open] <fridge> ({fridge_id})',
                    f'<char0> [putin] <{food}> ({food_id}) <fridge> ({fridge_id})',
                    f'<char0> [close] <fridge> ({fridge_id})',
                ]
                res = comm.render_script(script, recording=False, skip_animation=True)
                print(f'food {food}: {res}')

    comm.close()


if __name__ == '__main__':
    # find_food_test()
    # find_object_test()
    # find_character_relation_test()
    # operate_food_test()
    # object_properties_test()
    # operate_salmon_script_test()
    operate_food_script_test()
