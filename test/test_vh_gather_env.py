from src.vh_env import VirtualHomeGatherFoodEnv
from virtualhome.simulation.unity_simulator.comm_unity import UnityCommunication
from src.vh_env.enums import ActionEnum
from stable_baselines3.common.env_checker import check_env
import unittest

class VhGatherEnvTest(unittest.TestCase):
    # test virtual home gather food environment
    YOUR_FILE_NAME = "D:\\programs\\windows_exec.v2.2.4\\VirtualHome.exe"

    def setUp(self):
        self.comm = UnityCommunication(file_name=self.YOUR_FILE_NAME)

    def tearDown(self):
        self.comm.close()

    def util_put_food_in_fridge(self, vh_env: VirtualHomeGatherFoodEnv, food: str):

        food_index_dict = vh_env.get_food_index_dict()
        food_index = food_index_dict[food]
        object_index_dict = vh_env.get_object_index_dict()
        object_index = object_index_dict['fridge']

        vh_env.step(action=(ActionEnum.WALK_TO_FOOD, food_index, 0))  # '<char0> [walk] <salmon> (328)'
        vh_env.step(action=(ActionEnum.GRAB, food_index, 0))  # '<char0> [grab] <salmon> (328)'
        vh_env.step(action=(ActionEnum.WALK_TO_OBJECT, 0, object_index))  # '<char0> [walk] <fridge> (314)
        vh_env.step(action=(ActionEnum.OPEN, 0, object_index))  # '<char0> [walk] <fridge> (306)'
        vh_env.step(action=(ActionEnum.PUTIN, food_index, object_index))  # '<char0> [putin] <salmon> (328) <fridge> (306)'
        vh_env.step(action=(ActionEnum.CLOSE, 0, object_index))  # '<char0> [close] <fridge> (306)'

    def util_stop_epoch(self, vh_env: VirtualHomeGatherFoodEnv):
        vh_env.step(action=(ActionEnum.STOP, 0, 0))

    # def test_stablebaselines3_checkenv(self):
    #     check_env(vh_env, warn=True, skip_render_check=True)
    #     self.assertTrue(True)

    def test_t_env0_grab_food_one_by_one(self):
        self.comm.reset(0)
        self.comm.add_character()
        res, g = self.comm.environment_graph()
        vh_env = VirtualHomeGatherFoodEnv(environment_graph=g, log_level='debug')
        vh_env.reset()

        food_list = ['salmon','pie','chocolatesyrup']

        for food in food_list:
            self.util_put_food_in_fridge(vh_env, food)

        self.util_stop_epoch(vh_env)

        print(f'instruct sequence: {vh_env.get_instruction_list()}')

        self.assertTrue(True)

    def test_t_env0_grab_food_fullhands(self):
        self.comm.reset(0)
        self.comm.add_character()
        res, g = self.comm.environment_graph()
        vh_env = VirtualHomeGatherFoodEnv(environment_graph=g)
        vh_env.reset()

        food_index_dict = vh_env.get_food_index_dict()
        object_index_dict = vh_env.get_object_index_dict()
        object_index = object_index_dict['fridge']

        food_list = ['salmon','pie','chocolatesyrup']
        salmon_index = food_index_dict['salmon']
        pie_index = food_index_dict['pie']
        chocolatesyrup_index = food_index_dict['chocolatesyrup']

        vh_env.step(action=(ActionEnum.WALK_TO_FOOD, salmon_index, 0))  # '<char0> [walk] <salmon> (328)'
        vh_env.step(action=(ActionEnum.GRAB, salmon_index, 0))  # '<char0> [grab] <salmon> (328)'
        vh_env.step(action=(ActionEnum.WALK_TO_OBJECT, 0, object_index))  # '<char0> [walk] <fridge> (314)
        vh_env.step(action=(ActionEnum.OPEN, 0, object_index))  # '<char0> [walk] <fridge> (306)'
        vh_env.step(action=(ActionEnum.PUTIN, salmon_index, object_index))  # '<char0> [putin] <salmon> (328) <fridge> (306)'

        vh_env.step(action=(ActionEnum.WALK_TO_FOOD, pie_index, 0))
        vh_env.step(action=(ActionEnum.GRAB, pie_index, 0))
        vh_env.step(action=(ActionEnum.WALK_TO_FOOD, chocolatesyrup_index, 0))
        vh_env.step(action=(ActionEnum.GRAB, chocolatesyrup_index, 0))

        vh_env.step(action=(ActionEnum.WALK_TO_OBJECT, 0, object_index))  # '<char0> [walk] <fridge> (314)
        vh_env.step(action=(ActionEnum.PUTIN, pie_index, object_index))
        vh_env.step(action=(ActionEnum.PUTIN, chocolatesyrup_index, object_index))
        vh_env.step(action=(ActionEnum.CLOSE, 0, object_index))  # '<char0> [close] <fridge> (306)'

        vh_env.step(action=(ActionEnum.STOP, 0, 0))

        print(f'instruct sequence: {vh_env.get_instruction_list()}')
        self.assertTrue(True)

    def test_t_env0_put_food_on_coffeetable(self):
        self.comm.reset(0)
        self.comm.add_character()
        res, g = self.comm.environment_graph()
        vh_env = VirtualHomeGatherFoodEnv(environment_graph=g)
        vh_env.reset()

        food_index_dict = vh_env.get_food_index_dict()
        object_index_dict = vh_env.get_object_index_dict()
        object_index = object_index_dict['coffeetable']

        salmon_index = food_index_dict['salmon']
        pie_index = food_index_dict['pie']
        chocolatesyrup_index = food_index_dict['chocolatesyrup']

        vh_env.step(action=(ActionEnum.WALK_TO_FOOD, salmon_index, 0))
        vh_env.step(action=(ActionEnum.GRAB, salmon_index, 0))
        vh_env.step(action=(ActionEnum.WALK_TO_OBJECT, 0, object_index))
        vh_env.step(action=(ActionEnum.PUT, salmon_index, object_index))
        vh_env.step(action=(ActionEnum.STOP, 0, 0))
        print(f'instruct sequence: {vh_env.get_instruction_list()}')
        self.assertTrue(True)


    def test_t_env0_put_food_on_fridge(self):

        # Character can put food on fridge.
        self.comm.reset(0)
        self.comm.add_character()
        res, g = self.comm.environment_graph()
        vh_env = VirtualHomeGatherFoodEnv(environment_graph=g)
        vh_env.reset()

        food_index_dict = vh_env.get_food_index_dict()
        object_index_dict = vh_env.get_object_index_dict()
        object_index = object_index_dict['fridge']

        salmon_index = food_index_dict['salmon']
        pie_index = food_index_dict['pie']
        chocolatesyrup_index = food_index_dict['chocolatesyrup']

        vh_env.step(action=(ActionEnum.WALK_TO_FOOD, salmon_index, 0))
        vh_env.step(action=(ActionEnum.GRAB, salmon_index, 0))
        vh_env.step(action=(ActionEnum.WALK_TO_OBJECT, 0, object_index))
        tup = vh_env.step(action=(ActionEnum.PUT, salmon_index, object_index))
        vh_env.step(action=(ActionEnum.STOP, 0, 0))

        print(f'instruct sequence: {vh_env.get_instruction_list()}')
        self.assertTrue(True)

    def test_t_env0_grab_food_inside_fridge(self):
        self.comm.reset(0)
        self.comm.add_character()
        res, g = self.comm.environment_graph()
        vh_env = VirtualHomeGatherFoodEnv(environment_graph=g)
        vh_env.reset()

        food_index_dict = vh_env.get_food_index_dict()
        object_index_dict = vh_env.get_object_index_dict()
        object_index = object_index_dict['fridge']

        salmon_index = food_index_dict['salmon']
        pie_index = food_index_dict['pie']
        chocolatesyrup_index = food_index_dict['chocolatesyrup']

        vh_env.step(action=(ActionEnum.WALK_TO_FOOD, salmon_index, 0))
        vh_env.step(action=(ActionEnum.GRAB, salmon_index, 0))
        vh_env.step(action=(ActionEnum.WALK_TO_OBJECT, 0, object_index))
        vh_env.step(action=(ActionEnum.OPEN, 0, object_index))
        vh_env.step(action=(ActionEnum.PUTIN, salmon_index, object_index))
        tup = vh_env.step(action=(ActionEnum.GRAB, salmon_index, 0))
        vh_env.step(action=(ActionEnum.STOP, 0, 0))
        print(f'instruct sequence: {vh_env.get_instruction_list()}')
        self.assertTrue(float(tup[1]) < 0.)

    def test_f_env0_grab_2plus_food(self):
        self.comm.reset(0)
        self.comm.add_character()
        res, g = self.comm.environment_graph()
        vh_env = VirtualHomeGatherFoodEnv(environment_graph=g)
        vh_env.reset()

        food_index_dict = vh_env.get_food_index_dict()
        object_index_dict = vh_env.get_object_index_dict()
        object_index = object_index_dict['fridge']

        salmon_index = food_index_dict['salmon']
        pie_index = food_index_dict['pie']
        chocolatesyrup_index = food_index_dict['chocolatesyrup']

        vh_env.step(action=(ActionEnum.WALK_TO_FOOD, salmon_index, 0))  # '<char0> [walk] <salmon> (328)'
        vh_env.step(action=(ActionEnum.GRAB, salmon_index, 0))  # '<char0> [grab] <salmon> (328)'
        vh_env.step(action=(ActionEnum.WALK_TO_FOOD, pie_index, 0))
        vh_env.step(action=(ActionEnum.GRAB, pie_index, 0))
        vh_env.step(action=(ActionEnum.WALK_TO_FOOD, chocolatesyrup_index, 0))
        tup = vh_env.step(action=(ActionEnum.GRAB, chocolatesyrup_index, 0))
        vh_env.step(action=(ActionEnum.STOP, 0, 0))
        print(f'instruct sequence: {vh_env.get_instruction_list()}')
        self.assertTrue(float(tup[1]) < 0.)

    def test_f_env0_grab_food_faraway_food_1(self):
        self.comm.reset(0)
        self.comm.add_character()
        res, g = self.comm.environment_graph()
        vh_env = VirtualHomeGatherFoodEnv(environment_graph=g)
        vh_env.reset()

        food_index_dict = vh_env.get_food_index_dict()
        object_index_dict = vh_env.get_object_index_dict()
        object_index = object_index_dict['fridge']

        salmon_index = food_index_dict['salmon']
        pie_index = food_index_dict['pie']
        chocolatesyrup_index = food_index_dict['chocolatesyrup']

        vh_env.step(action=(ActionEnum.WALK_TO_FOOD, salmon_index, 0))
        tup = vh_env.step(action=(ActionEnum.GRAB, pie_index, 0))
        vh_env.step(action=(ActionEnum.STOP, 0, 0))
        print(f'instruct sequence: {vh_env.get_instruction_list()}')
        self.assertTrue(float(tup[1]) < 0.)

    def test_f_env0_grab_food_faraway_food_2(self):
        self.comm.reset(0)
        self.comm.add_character()
        res, g = self.comm.environment_graph()
        vh_env = VirtualHomeGatherFoodEnv(environment_graph=g)
        vh_env.reset()

        food_index_dict = vh_env.get_food_index_dict()
        object_index_dict = vh_env.get_object_index_dict()
        object_index = object_index_dict['fridge']

        salmon_index = food_index_dict['salmon']
        pie_index = food_index_dict['pie']
        chocolatesyrup_index = food_index_dict['chocolatesyrup']

        vh_env.step(action=(ActionEnum.WALK_TO_OBJECT, 0, object_index))
        tup = vh_env.step(action=(ActionEnum.GRAB, pie_index, 0))
        vh_env.step(action=(ActionEnum.STOP, 0, 0))
        print(f'instruct sequence: {vh_env.get_instruction_list()}')
        self.assertTrue(float(tup[1]) < 0.)

    def test_f_env0_grab_food_inside_closed_object(self):
        self.comm.reset(0)
        self.comm.add_character()
        res, g = self.comm.environment_graph()
        vh_env = VirtualHomeGatherFoodEnv(environment_graph=g)
        vh_env.reset()

        food_index_dict = vh_env.get_food_index_dict()
        object_index_dict = vh_env.get_object_index_dict()
        object_index = object_index_dict['fridge']

        salmon_index = food_index_dict['salmon']
        pie_index = food_index_dict['pie']
        chocolatesyrup_index = food_index_dict['chocolatesyrup']

        vh_env.step(action=(ActionEnum.WALK_TO_FOOD, salmon_index, 0))
        vh_env.step(action=(ActionEnum.GRAB, salmon_index, 0))
        vh_env.step(action=(ActionEnum.WALK_TO_OBJECT, 0, object_index))
        vh_env.step(action=(ActionEnum.OPEN, 0 , object_index))
        vh_env.step(action=(ActionEnum.PUTIN, salmon_index, object_index))
        vh_env.step(action=(ActionEnum.CLOSE, 0, object_index))
        tup = vh_env.step(action=(ActionEnum.GRAB, salmon_index, 0))
        vh_env.step(action=(ActionEnum.STOP, 0, 0))
        print(f'instruct sequence: {vh_env.get_instruction_list()}')
        self.assertTrue(float(tup[1]) < 0.)

    def test_f_env0_put_food_without_grabbed(self):
        self.comm.reset(0)
        self.comm.add_character()
        res, g = self.comm.environment_graph()
        vh_env = VirtualHomeGatherFoodEnv(environment_graph=g)
        vh_env.reset()

        food_index_dict = vh_env.get_food_index_dict()
        object_index_dict = vh_env.get_object_index_dict()
        object_index = object_index_dict['fridge']

        salmon_index = food_index_dict['salmon']
        pie_index = food_index_dict['pie']
        chocolatesyrup_index = food_index_dict['chocolatesyrup']

        vh_env.step(action=(ActionEnum.WALK_TO_OBJECT, 0, object_index))
        tup = vh_env.step(action=(ActionEnum.PUT, salmon_index, object_index))
        vh_env.step(action=(ActionEnum.STOP, 0, 0))
        print(f'instruct sequence: {vh_env.get_instruction_list()}')
        self.assertTrue(float(tup[1]) < 0.)

    def test_f_env0_put_food_to_faraway_object(self):

        self.comm.reset(0)
        self.comm.add_character()
        res, g = self.comm.environment_graph()
        vh_env = VirtualHomeGatherFoodEnv(environment_graph=g)
        vh_env.reset()

        food_index_dict = vh_env.get_food_index_dict()
        object_index_dict = vh_env.get_object_index_dict()
        object_index = object_index_dict['fridge']

        salmon_index = food_index_dict['salmon']
        pie_index = food_index_dict['pie']
        chocolatesyrup_index = food_index_dict['chocolatesyrup']

        vh_env.step(action=(ActionEnum.WALK_TO_FOOD, salmon_index, 0))
        vh_env.step(action=(ActionEnum.GRAB, salmon_index, 0))
        # vh_env.step(action=(ActionEnum.WALK_TO_OBJECT, 0, object_index))
        tup = vh_env.step(action=(ActionEnum.PUT, salmon_index, object_index))
        vh_env.step(action=(ActionEnum.STOP, 0, 0))

        print(f'instruct sequence: {vh_env.get_instruction_list()}')
        self.assertTrue(float(tup[1]) < 0.)

    def test_f_env0_putin_food_without_grabbed(self):
        self.comm.reset(0)
        self.comm.add_character()
        res, g = self.comm.environment_graph()
        vh_env = VirtualHomeGatherFoodEnv(environment_graph=g)
        vh_env.reset()

        food_index_dict = vh_env.get_food_index_dict()
        object_index_dict = vh_env.get_object_index_dict()
        object_index = object_index_dict['fridge']

        salmon_index = food_index_dict['salmon']
        pie_index = food_index_dict['pie']
        chocolatesyrup_index = food_index_dict['chocolatesyrup']

        vh_env.step(action=(ActionEnum.WALK_TO_OBJECT, 0, object_index))
        tup = vh_env.step(action=(ActionEnum.PUTIN, salmon_index, object_index))
        vh_env.step(action=(ActionEnum.STOP, 0, 0))
        print(f'instruct sequence: {vh_env.get_instruction_list()}')
        self.assertTrue(float(tup[1]) < 0.)

    def test_f_env0_putin_food_to_faraway_object(self):

        self.comm.reset(0)
        self.comm.add_character()
        res, g = self.comm.environment_graph()
        vh_env = VirtualHomeGatherFoodEnv(environment_graph=g)
        vh_env.reset()

        food_index_dict = vh_env.get_food_index_dict()
        object_index_dict = vh_env.get_object_index_dict()
        object_index = object_index_dict['fridge']

        salmon_index = food_index_dict['salmon']
        pie_index = food_index_dict['pie']
        chocolatesyrup_index = food_index_dict['chocolatesyrup']

        vh_env.step(action=(ActionEnum.WALK_TO_FOOD, salmon_index, 0))
        vh_env.step(action=(ActionEnum.GRAB, salmon_index, 0))
        tup = vh_env.step(action=(ActionEnum.PUTIN, salmon_index, object_index))
        vh_env.step(action=(ActionEnum.STOP, 0, 0))

        print(f'instruct sequence: {vh_env.get_instruction_list()}')
        self.assertTrue(float(tup[1]) < 0.)

    def test_f_env0_putin_food_to_object_not_openable(self):
        self.comm.reset(0)
        self.comm.add_character()
        res, g = self.comm.environment_graph()
        vh_env = VirtualHomeGatherFoodEnv(environment_graph=g)
        vh_env.reset()

        food_index_dict = vh_env.get_food_index_dict()
        object_index_dict = vh_env.get_object_index_dict()
        object_index = object_index_dict['coffeetable']

        salmon_index = food_index_dict['salmon']
        pie_index = food_index_dict['pie']
        chocolatesyrup_index = food_index_dict['chocolatesyrup']

        vh_env.step(action=(ActionEnum.WALK_TO_FOOD, salmon_index, 0))
        vh_env.step(action=(ActionEnum.GRAB, salmon_index, 0))
        vh_env.step(action=(ActionEnum.WALK_TO_OBJECT, 0, object_index))
        tup = vh_env.step(action=(ActionEnum.PUTIN, salmon_index, object_index))
        vh_env.step(action=(ActionEnum.STOP, 0, 0))

        print(f'instruct sequence: {vh_env.get_instruction_list()}')
        self.assertTrue(float(tup[1]) < 0.)

    def test_f_env0_open_object_not_openable(self):
        self.comm.reset(0)
        self.comm.add_character()
        res, g = self.comm.environment_graph()
        vh_env = VirtualHomeGatherFoodEnv(environment_graph=g)
        vh_env.reset()


        food_index_dict = vh_env.get_food_index_dict()
        object_index_dict = vh_env.get_object_index_dict()
        object_index = object_index_dict['coffeetable']

        salmon_index = food_index_dict['salmon']
        pie_index = food_index_dict['pie']
        chocolatesyrup_index = food_index_dict['chocolatesyrup']

        vh_env.step(action=(ActionEnum.WALK_TO_FOOD, salmon_index, 0))
        vh_env.step(action=(ActionEnum.GRAB, salmon_index, 0))
        vh_env.step(action=(ActionEnum.WALK_TO_OBJECT, 0, object_index))
        tup = vh_env.step(action=(ActionEnum.OPEN, 0, object_index))
        vh_env.step(action=(ActionEnum.STOP, 0, 0))

        print(f'instruct sequence: {vh_env.get_instruction_list()}')
        self.assertTrue(float(tup[1]) < 0.)

    def test_f_env0_open_faraway_object(self):
        self.comm.reset(0)
        self.comm.add_character()
        res, g = self.comm.environment_graph()
        vh_env = VirtualHomeGatherFoodEnv(environment_graph=g)
        vh_env.reset()

        food_index_dict = vh_env.get_food_index_dict()
        object_index_dict = vh_env.get_object_index_dict()
        object_index = object_index_dict['fridge']

        salmon_index = food_index_dict['salmon']
        pie_index = food_index_dict['pie']
        chocolatesyrup_index = food_index_dict['chocolatesyrup']


        vh_env.step(action=(ActionEnum.WALK_TO_FOOD, salmon_index, 0))
        vh_env.step(action=(ActionEnum.GRAB, salmon_index, 0))
        # vh_env.step(action=(ActionEnum.WALK_TO_OBJECT, 0, object_index))
        tup = vh_env.step(action=(ActionEnum.OPEN, 0, object_index))
        vh_env.step(action=(ActionEnum.STOP, 0, 0))

        print(f'instruct sequence: {vh_env.get_instruction_list()}')
        self.assertTrue(float(tup[1]) < 0.)

    def test_f_env0_open_object_already_opened(self):
        self.comm.reset(0)
        self.comm.add_character()
        res, g = self.comm.environment_graph()
        vh_env = VirtualHomeGatherFoodEnv(environment_graph=g)
        vh_env.reset()

        food_index_dict = vh_env.get_food_index_dict()
        object_index_dict = vh_env.get_object_index_dict()
        object_index = object_index_dict['fridge']

        salmon_index = food_index_dict['salmon']
        pie_index = food_index_dict['pie']
        chocolatesyrup_index = food_index_dict['chocolatesyrup']


        vh_env.step(action=(ActionEnum.WALK_TO_FOOD, salmon_index, 0))
        vh_env.step(action=(ActionEnum.GRAB, salmon_index, 0))
        vh_env.step(action=(ActionEnum.WALK_TO_OBJECT, 0, object_index))
        vh_env.step(action=(ActionEnum.OPEN, 0, object_index))
        tup = vh_env.step(action=(ActionEnum.OPEN, 0, object_index))
        vh_env.step(action=(ActionEnum.STOP, 0, 0))

        print(f'instruct sequence: {vh_env.get_instruction_list()}')
        self.assertTrue(float(tup[1]) < 0.)

    def test_f_env0_close_object_not_openable(self):
        self.comm.reset(0)
        self.comm.add_character()
        res, g = self.comm.environment_graph()
        vh_env = VirtualHomeGatherFoodEnv(environment_graph=g)
        vh_env.reset()

        food_index_dict = vh_env.get_food_index_dict()
        object_index_dict = vh_env.get_object_index_dict()
        object_index = object_index_dict['coffeetable']

        salmon_index = food_index_dict['salmon']
        pie_index = food_index_dict['pie']
        chocolatesyrup_index = food_index_dict['chocolatesyrup']

        vh_env.step(action=(ActionEnum.WALK_TO_OBJECT, 0, object_index))
        vh_env.step(action=(ActionEnum.OPEN, 0, object_index))
        tup = vh_env.step(action=(ActionEnum.CLOSE, 0, object_index))
        vh_env.step(action=(ActionEnum.STOP, 0, 0))

        print(f'instruct sequence: {vh_env.get_instruction_list()}')
        self.assertTrue(float(tup[1]) < 0.)

    def test_f_env0_close_object_already_closed(self):
        self.comm.reset(0)
        self.comm.add_character()
        res, g = self.comm.environment_graph()
        vh_env = VirtualHomeGatherFoodEnv(environment_graph=g)
        vh_env.reset()

        food_index_dict = vh_env.get_food_index_dict()
        object_index_dict = vh_env.get_object_index_dict()
        object_index = object_index_dict['fridge']

        salmon_index = food_index_dict['salmon']
        pie_index = food_index_dict['pie']
        chocolatesyrup_index = food_index_dict['chocolatesyrup']

        vh_env.step(action=(ActionEnum.WALK_TO_OBJECT, 0, object_index))
        vh_env.step(action=(ActionEnum.OPEN, 0, object_index))
        vh_env.step(action=(ActionEnum.CLOSE, 0, object_index))
        tup = vh_env.step(action=(ActionEnum.CLOSE, 0, object_index))
        vh_env.step(action=(ActionEnum.STOP, 0, 0))

        print(f'instruct sequence: {vh_env.get_instruction_list()}')
        self.assertTrue(float(tup[1]) < 0.)

    def test_f_env0_close_faraway_object(self):
        self.comm.reset(0)
        self.comm.add_character()
        res, g = self.comm.environment_graph()
        vh_env = VirtualHomeGatherFoodEnv(environment_graph=g)
        vh_env.reset()

        food_index_dict = vh_env.get_food_index_dict()
        object_index_dict = vh_env.get_object_index_dict()
        object_index = object_index_dict['fridge']

        salmon_index = food_index_dict['salmon']
        pie_index = food_index_dict['pie']
        chocolatesyrup_index = food_index_dict['chocolatesyrup']

        vh_env.step(action=(ActionEnum.WALK_TO_OBJECT, 0, object_index))
        vh_env.step(action=(ActionEnum.OPEN, 0, object_index))
        vh_env.step(action=(ActionEnum.WALK_TO_FOOD, salmon_index, 0))
        vh_env.step(action=(ActionEnum.GRAB, salmon_index, 0))
        tup = vh_env.step(action=(ActionEnum.CLOSE, 0, object_index))
        vh_env.step(action=(ActionEnum.STOP, 0, 0))

        print(f'instruct sequence: {vh_env.get_instruction_list()}')
        self.assertTrue(float(tup[1]) < 0.)

    def test_t_env1_grab_food_one_by_one(self):
        self.comm.reset(1)
        self.comm.add_character()
        res, g = self.comm.environment_graph()
        vh_env = VirtualHomeGatherFoodEnv(environment_graph=g)
        vh_env.reset()

        food_list = ['chicken','chocolatesyrup']

        for food in food_list:
            self.util_put_food_in_fridge(vh_env, food)

        self.util_stop_epoch(vh_env)

        print(f'instruct sequence: {vh_env.get_instruction_list()}')

        self.assertTrue(True)

    def test_t_env1_grab_chicken(self):
        self.comm.reset(1)
        self.comm.add_character()
        res, g = self.comm.environment_graph()
        vh_env = VirtualHomeGatherFoodEnv(environment_graph=g)
        vh_env.reset()

        food_index_dict = vh_env.get_food_index_dict()
        object_index_dict = vh_env.get_object_index_dict()

        chicken_index = food_index_dict['chicken']
        chocolatesyrup_index = food_index_dict['chocolatesyrup']

        fridge_index = object_index_dict['fridge']

        vh_env.step(action=(ActionEnum.WALK_TO_FOOD, chicken_index, 0))
        vh_env.step(action=(ActionEnum.GRAB, chicken_index, 0))
        vh_env.step(action=(ActionEnum.WALK_TO_OBJECT, 0, fridge_index))
        vh_env.step(action=(ActionEnum.OPEN, 0, fridge_index))
        vh_env.step(action=(ActionEnum.PUTIN, chicken_index, fridge_index))
        vh_env.step(action=(ActionEnum.CLOSE, 0, fridge_index))
        vh_env.step(action=(ActionEnum.STOP, 0, 0))

        print(f'instruct sequence: {vh_env.get_instruction_list()}')
        self.assertTrue(True)


    def test_f_env1_open_object_with_no_freehand(self):
        self.comm.reset(1)
        self.comm.add_character()
        res, g = self.comm.environment_graph()
        vh_env = VirtualHomeGatherFoodEnv(environment_graph=g)
        vh_env.reset()

        food_index_dict = vh_env.get_food_index_dict()
        object_index_dict = vh_env.get_object_index_dict()

        chicken_index = food_index_dict['chicken']
        chocolatesyrup_index = food_index_dict['chocolatesyrup']

        fridge_index = object_index_dict['fridge']

        vh_env.step(action=(ActionEnum.WALK_TO_FOOD, chicken_index, 0))
        vh_env.step(action=(ActionEnum.GRAB, chicken_index, 0))
        vh_env.step(action=(ActionEnum.WALK_TO_FOOD, chocolatesyrup_index, 0))
        vh_env.step(action=(ActionEnum.GRAB, chocolatesyrup_index, 0))
        vh_env.step(action=(ActionEnum.WALK_TO_OBJECT, 0, fridge_index))
        tup = vh_env.step(action=(ActionEnum.OPEN, 0, fridge_index))
        vh_env.step(action=(ActionEnum.STOP, 0, 0))

        print(f'instruct sequence: {vh_env.get_instruction_list()}')
        self.assertTrue(float(tup[1]) < 0.)

    def test_t_env2_grab_food_one_by_one(self):
        self.comm.reset(2)
        self.comm.add_character()
        res, g = self.comm.environment_graph()
        vh_env = VirtualHomeGatherFoodEnv(environment_graph=g)
        vh_env.reset()

        food_list = ['salmon','mincedmeat','juice']

        for food in food_list:
            self.util_put_food_in_fridge(vh_env, food)

        self.util_stop_epoch(vh_env)

        print(f'instruct sequence: {vh_env.get_instruction_list()}')

        self.assertTrue(True)


    def test_t_env3_grab_food_one_by_one(self):
        self.comm.reset(3)
        self.comm.add_character()
        res, g = self.comm.environment_graph()
        vh_env = VirtualHomeGatherFoodEnv(environment_graph=g)
        vh_env.reset()

        food_list = ['pancake','pear','milkshake', 'chocolatesyrup']

        for food in food_list:
            self.util_put_food_in_fridge(vh_env, food)

        self.util_stop_epoch(vh_env)

        print(f'instruct sequence: {vh_env.get_instruction_list()}')

        self.assertTrue(True)

    def test_t_env4_grab_food_one_by_one(self):
        self.comm.reset(4)
        self.comm.add_character()
        res, g = self.comm.environment_graph()
        vh_env = VirtualHomeGatherFoodEnv(environment_graph=g)
        vh_env.reset()

        food_list = ['pie','mincedmeat','chocolatesyrup']

        for food in food_list:
            self.util_put_food_in_fridge(vh_env, food)

        self.util_stop_epoch(vh_env)

        print(f'instruct sequence: {vh_env.get_instruction_list()}')

        self.assertTrue(True)

    def test_t_env5_grab_food(self):
        self.comm.reset(5)
        self.comm.add_character()
        res, g = self.comm.environment_graph()
        vh_env = VirtualHomeGatherFoodEnv(environment_graph=g)
        vh_env.reset()

        food_list = ['salad','carrot']

        for food in food_list:
            self.util_put_food_in_fridge(vh_env, food)

        self.util_stop_epoch(vh_env)

        print(f'instruct sequence: {vh_env.get_instruction_list()}')

        self.assertTrue(True)

    def test_t_env6_grab_food_one_by_one(self):
        self.comm.reset(6)
        self.comm.add_character()
        res, g = self.comm.environment_graph()
        vh_env = VirtualHomeGatherFoodEnv(environment_graph=g)
        vh_env.reset()

        food_list = ['wine','juice','chocolatesyrup', 'chicken']

        for food in food_list:
            self.util_put_food_in_fridge(vh_env, food)

        self.util_stop_epoch(vh_env)

        print(f'instruct sequence: {vh_env.get_instruction_list()}')

        self.assertTrue(True)


if __name__ == '__main__':
    unittest.main()


