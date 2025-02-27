from src.sb3vh.vh_env import VirtualHomeGatherFoodEnv
from virtualhome.simulation.unity_simulator.comm_unity import UnityCommunication
from src.sb3vh.env_graph_enum import ActionEnum
from stable_baselines3.common.env_checker import check_env
import unittest

class VhGatherEnvTest(unittest.TestCase):
    # test virtual home gather food environment
    YOUR_FILE_NAME = "D:\\programs\\windows_exec.v2.2.4\\VirtualHome.exe"

    def setUp(self):
        self.comm = UnityCommunication(file_name=self.YOUR_FILE_NAME)
        self.vh_env = VirtualHomeGatherFoodEnv(self.comm)

    def tearDown(self):
        self.vh_env.close()

    def util_put_food_in_fridge(self, food: str):

        food_index_dict = self.vh_env.get_food_index_dict()
        food_index = food_index_dict[food]
        object_index_dict = self.vh_env.get_object_index_dict()
        object_index = object_index_dict['fridge']

        self.vh_env.step(action=(ActionEnum.WALK_TO_FOOD, food_index, 0))  # '<char0> [walk] <salmon> (328)'
        self.vh_env.step(action=(ActionEnum.GRAB, food_index, 0))  # '<char0> [grab] <salmon> (328)'
        self.vh_env.step(action=(ActionEnum.WALK_TO_OBJECT, 0, object_index))  # '<char0> [walk] <fridge> (314)
        self.vh_env.step(action=(ActionEnum.OPEN, 0, object_index))  # '<char0> [walk] <fridge> (306)'
        self.vh_env.step(action=(ActionEnum.PUTIN, food_index, object_index))  # '<char0> [putin] <salmon> (328) <fridge> (306)'
        self.vh_env.step(action=(ActionEnum.CLOSE, 0, object_index))  # '<char0> [close] <fridge> (306)'

    def util_stop_epoch(self):
        self.vh_env.step(action=(ActionEnum.STOP, 0, 0))

    # def test_stablebaselines3_checkenv(self):
    #     check_env(self.vh_env, warn=True, skip_render_check=True)
    #     self.assertTrue(True)

    def test_t_env0_grab_food_one_by_one(self):
        self.vh_env.reset(options={
            'environment_index': 0,
        })

        food_list = ['salmon','pie','chocolatesyrup']

        for food in food_list:
            self.util_put_food_in_fridge(food)

        self.util_stop_epoch()

        print(f'instruct sequence: {self.vh_env.print_instruct()}')

        self.assertTrue(True)

    def test_t_env0_grab_food_fullhands(self):
        self.vh_env.reset(options={
            'environment_index': 0,
        })
        food_index_dict = self.vh_env.get_food_index_dict()
        object_index_dict = self.vh_env.get_object_index_dict()
        object_index = object_index_dict['fridge']

        food_list = ['salmon','pie','chocolatesyrup']
        salmon_index = food_index_dict['salmon']
        pie_index = food_index_dict['pie']
        chocolatesyrup_index = food_index_dict['chocolatesyrup']

        self.vh_env.step(action=(ActionEnum.WALK_TO_FOOD, salmon_index, 0))  # '<char0> [walk] <salmon> (328)'
        self.vh_env.step(action=(ActionEnum.GRAB, salmon_index, 0))  # '<char0> [grab] <salmon> (328)'
        self.vh_env.step(action=(ActionEnum.WALK_TO_OBJECT, 0, object_index))  # '<char0> [walk] <fridge> (314)
        self.vh_env.step(action=(ActionEnum.OPEN, 0, object_index))  # '<char0> [walk] <fridge> (306)'
        self.vh_env.step(action=(ActionEnum.PUTIN, salmon_index, object_index))  # '<char0> [putin] <salmon> (328) <fridge> (306)'

        self.vh_env.step(action=(ActionEnum.WALK_TO_FOOD, pie_index, 0))
        self.vh_env.step(action=(ActionEnum.GRAB, pie_index, 0))
        self.vh_env.step(action=(ActionEnum.WALK_TO_FOOD, chocolatesyrup_index, 0))
        self.vh_env.step(action=(ActionEnum.GRAB, chocolatesyrup_index, 0))

        self.vh_env.step(action=(ActionEnum.WALK_TO_OBJECT, 0, object_index))  # '<char0> [walk] <fridge> (314)
        self.vh_env.step(action=(ActionEnum.PUTIN, pie_index, object_index))
        self.vh_env.step(action=(ActionEnum.PUTIN, chocolatesyrup_index, object_index))
        self.vh_env.step(action=(ActionEnum.CLOSE, 0, object_index))  # '<char0> [close] <fridge> (306)'

        self.vh_env.step(action=(ActionEnum.STOP, 0, 0))

        print(f'instruct sequence: {self.vh_env.print_instruct()}')
        self.assertTrue(True)

    def test_t_env0_put_food_on_coffeetable(self):
        self.vh_env.reset(options={
            'environment_index': 0,
        })
        food_index_dict = self.vh_env.get_food_index_dict()
        object_index_dict = self.vh_env.get_object_index_dict()
        object_index = object_index_dict['coffeetable']

        salmon_index = food_index_dict['salmon']
        pie_index = food_index_dict['pie']
        chocolatesyrup_index = food_index_dict['chocolatesyrup']

        self.vh_env.step(action=(ActionEnum.WALK_TO_FOOD, salmon_index, 0))
        self.vh_env.step(action=(ActionEnum.GRAB, salmon_index, 0))
        self.vh_env.step(action=(ActionEnum.WALK_TO_OBJECT, 0, object_index))
        self.vh_env.step(action=(ActionEnum.PUT, salmon_index, object_index))
        self.vh_env.step(action=(ActionEnum.STOP, 0, 0))
        print(f'instruct sequence: {self.vh_env.print_instruct()}')
        self.assertTrue(True)


    def test_t_env0_put_food_on_fridge(self):

        # Character can put food on fridge.
        self.vh_env.reset(options={
            'environment_index': 0,
        })
        food_index_dict = self.vh_env.get_food_index_dict()
        object_index_dict = self.vh_env.get_object_index_dict()
        object_index = object_index_dict['fridge']

        salmon_index = food_index_dict['salmon']
        pie_index = food_index_dict['pie']
        chocolatesyrup_index = food_index_dict['chocolatesyrup']

        self.vh_env.step(action=(ActionEnum.WALK_TO_FOOD, salmon_index, 0))
        self.vh_env.step(action=(ActionEnum.GRAB, salmon_index, 0))
        self.vh_env.step(action=(ActionEnum.WALK_TO_OBJECT, 0, object_index))
        tup = self.vh_env.step(action=(ActionEnum.PUT, salmon_index, object_index))
        self.vh_env.step(action=(ActionEnum.STOP, 0, 0))

        print(f'instruct sequence: {self.vh_env.print_instruct()}')
        self.assertTrue(True)

    def test_t_env0_grab_food_inside_fridge(self):
        self.vh_env.reset(options={
            'environment_index': 0,
        })
        food_index_dict = self.vh_env.get_food_index_dict()
        object_index_dict = self.vh_env.get_object_index_dict()
        object_index = object_index_dict['fridge']

        salmon_index = food_index_dict['salmon']
        pie_index = food_index_dict['pie']
        chocolatesyrup_index = food_index_dict['chocolatesyrup']

        self.vh_env.step(action=(ActionEnum.WALK_TO_FOOD, salmon_index, 0))
        self.vh_env.step(action=(ActionEnum.GRAB, salmon_index, 0))
        self.vh_env.step(action=(ActionEnum.WALK_TO_OBJECT, 0, object_index))
        self.vh_env.step(action=(ActionEnum.OPEN, 0, object_index))
        self.vh_env.step(action=(ActionEnum.PUTIN, salmon_index, object_index))
        tup = self.vh_env.step(action=(ActionEnum.GRAB, salmon_index, 0))
        self.vh_env.step(action=(ActionEnum.STOP, 0, 0))
        print(f'instruct sequence: {self.vh_env.print_instruct()}')
        self.assertTrue(float(tup[1]) < 0.)

    def test_f_env0_grab_2plus_food(self):
        self.vh_env.reset(options={
            'environment_index': 0,
        })
        food_index_dict = self.vh_env.get_food_index_dict()
        object_index_dict = self.vh_env.get_object_index_dict()
        object_index = object_index_dict['fridge']

        salmon_index = food_index_dict['salmon']
        pie_index = food_index_dict['pie']
        chocolatesyrup_index = food_index_dict['chocolatesyrup']

        self.vh_env.step(action=(ActionEnum.WALK_TO_FOOD, salmon_index, 0))  # '<char0> [walk] <salmon> (328)'
        self.vh_env.step(action=(ActionEnum.GRAB, salmon_index, 0))  # '<char0> [grab] <salmon> (328)'
        self.vh_env.step(action=(ActionEnum.WALK_TO_FOOD, pie_index, 0))
        self.vh_env.step(action=(ActionEnum.GRAB, pie_index, 0))
        self.vh_env.step(action=(ActionEnum.WALK_TO_FOOD, chocolatesyrup_index, 0))
        tup = self.vh_env.step(action=(ActionEnum.GRAB, chocolatesyrup_index, 0))
        self.vh_env.step(action=(ActionEnum.STOP, 0, 0))
        print(f'instruct sequence: {self.vh_env.print_instruct()}')
        self.assertTrue(float(tup[1]) < 0.)
        
    def test_f_env0_grab_food_faraway_food_1(self):
        self.vh_env.reset(options={
            'environment_index': 0,
        })
        food_index_dict = self.vh_env.get_food_index_dict()
        object_index_dict = self.vh_env.get_object_index_dict()
        object_index = object_index_dict['fridge']
        
        salmon_index = food_index_dict['salmon']
        pie_index = food_index_dict['pie']
        chocolatesyrup_index = food_index_dict['chocolatesyrup']
        
        self.vh_env.step(action=(ActionEnum.WALK_TO_FOOD, salmon_index, 0)) 
        tup = self.vh_env.step(action=(ActionEnum.GRAB, pie_index, 0))
        self.vh_env.step(action=(ActionEnum.STOP, 0, 0))
        print(f'instruct sequence: {self.vh_env.print_instruct()}')
        self.assertTrue(float(tup[1]) < 0.)

    def test_f_env0_grab_food_faraway_food_2(self):
        self.vh_env.reset(options={
            'environment_index': 0,
        })
        food_index_dict = self.vh_env.get_food_index_dict()
        object_index_dict = self.vh_env.get_object_index_dict()
        object_index = object_index_dict['fridge']

        salmon_index = food_index_dict['salmon']
        pie_index = food_index_dict['pie']
        chocolatesyrup_index = food_index_dict['chocolatesyrup']

        self.vh_env.step(action=(ActionEnum.WALK_TO_OBJECT, 0, object_index))
        tup = self.vh_env.step(action=(ActionEnum.GRAB, pie_index, 0))
        self.vh_env.step(action=(ActionEnum.STOP, 0, 0))
        print(f'instruct sequence: {self.vh_env.print_instruct()}')
        self.assertTrue(float(tup[1]) < 0.)

    def test_f_env0_grab_food_inside_closed_object(self):
        self.vh_env.reset(options={
            'environment_index': 0,
        })
        food_index_dict = self.vh_env.get_food_index_dict()
        object_index_dict = self.vh_env.get_object_index_dict()
        object_index = object_index_dict['fridge']

        salmon_index = food_index_dict['salmon']
        pie_index = food_index_dict['pie']
        chocolatesyrup_index = food_index_dict['chocolatesyrup']

        self.vh_env.step(action=(ActionEnum.WALK_TO_FOOD, salmon_index, 0))
        self.vh_env.step(action=(ActionEnum.GRAB, salmon_index, 0))
        self.vh_env.step(action=(ActionEnum.WALK_TO_OBJECT, 0, object_index))
        self.vh_env.step(action=(ActionEnum.OPEN, 0 , object_index))
        self.vh_env.step(action=(ActionEnum.PUTIN, salmon_index, object_index))
        self.vh_env.step(action=(ActionEnum.CLOSE, 0, object_index))
        tup = self.vh_env.step(action=(ActionEnum.GRAB, salmon_index, 0))
        self.vh_env.step(action=(ActionEnum.STOP, 0, 0))
        print(f'instruct sequence: {self.vh_env.print_instruct()}')
        self.assertTrue(float(tup[1]) < 0.)
    
    def test_f_env0_put_food_without_grabbed(self):
        self.vh_env.reset(options={
            'environment_index': 0,
        })
        food_index_dict = self.vh_env.get_food_index_dict()
        object_index_dict = self.vh_env.get_object_index_dict()
        object_index = object_index_dict['fridge']

        salmon_index = food_index_dict['salmon']
        pie_index = food_index_dict['pie']
        chocolatesyrup_index = food_index_dict['chocolatesyrup']

        self.vh_env.step(action=(ActionEnum.WALK_TO_OBJECT, 0, object_index))
        tup = self.vh_env.step(action=(ActionEnum.PUT, salmon_index, object_index))
        self.vh_env.step(action=(ActionEnum.STOP, 0, 0))
        print(f'instruct sequence: {self.vh_env.print_instruct()}')
        self.assertTrue(float(tup[1]) < 0.)

    def test_f_env0_put_food_to_faraway_object(self):

        # Character can put food on fridge.
        self.vh_env.reset(options={
            'environment_index': 0,
        })
        food_index_dict = self.vh_env.get_food_index_dict()
        object_index_dict = self.vh_env.get_object_index_dict()
        object_index = object_index_dict['fridge']

        salmon_index = food_index_dict['salmon']
        pie_index = food_index_dict['pie']
        chocolatesyrup_index = food_index_dict['chocolatesyrup']

        self.vh_env.step(action=(ActionEnum.WALK_TO_FOOD, salmon_index, 0))
        self.vh_env.step(action=(ActionEnum.GRAB, salmon_index, 0))
        # self.vh_env.step(action=(ActionEnum.WALK_TO_OBJECT, 0, object_index))
        tup = self.vh_env.step(action=(ActionEnum.PUT, salmon_index, object_index))
        self.vh_env.step(action=(ActionEnum.STOP, 0, 0))

        print(f'instruct sequence: {self.vh_env.print_instruct()}')
        self.assertTrue(float(tup[1]) < 0.)

    def test_f_env0_putin_food_without_grabbed(self):
        self.vh_env.reset(options={
            'environment_index': 0,
        })
        food_index_dict = self.vh_env.get_food_index_dict()
        object_index_dict = self.vh_env.get_object_index_dict()
        object_index = object_index_dict['fridge']

        salmon_index = food_index_dict['salmon']
        pie_index = food_index_dict['pie']
        chocolatesyrup_index = food_index_dict['chocolatesyrup']

        self.vh_env.step(action=(ActionEnum.WALK_TO_OBJECT, 0, object_index))
        tup = self.vh_env.step(action=(ActionEnum.PUTIN, salmon_index, object_index))
        self.vh_env.step(action=(ActionEnum.STOP, 0, 0))
        print(f'instruct sequence: {self.vh_env.print_instruct()}')
        self.assertTrue(float(tup[1]) < 0.)

    def test_f_env0_putin_food_to_faraway_object(self):

        self.vh_env.reset(options={
            'environment_index': 0,
        })
        food_index_dict = self.vh_env.get_food_index_dict()
        object_index_dict = self.vh_env.get_object_index_dict()
        object_index = object_index_dict['fridge']

        salmon_index = food_index_dict['salmon']
        pie_index = food_index_dict['pie']
        chocolatesyrup_index = food_index_dict['chocolatesyrup']

        self.vh_env.step(action=(ActionEnum.WALK_TO_FOOD, salmon_index, 0))
        self.vh_env.step(action=(ActionEnum.GRAB, salmon_index, 0))
        tup = self.vh_env.step(action=(ActionEnum.PUTIN, salmon_index, object_index))
        self.vh_env.step(action=(ActionEnum.STOP, 0, 0))

        print(f'instruct sequence: {self.vh_env.print_instruct()}')
        self.assertTrue(float(tup[1]) < 0.)

    def test_f_env0_putin_food_to_object_not_openable(self):
        self.vh_env.reset(options={
            'environment_index': 0,
        })
        food_index_dict = self.vh_env.get_food_index_dict()
        object_index_dict = self.vh_env.get_object_index_dict()
        object_index = object_index_dict['coffeetable']

        salmon_index = food_index_dict['salmon']
        pie_index = food_index_dict['pie']
        chocolatesyrup_index = food_index_dict['chocolatesyrup']

        self.vh_env.step(action=(ActionEnum.WALK_TO_FOOD, salmon_index, 0))
        self.vh_env.step(action=(ActionEnum.GRAB, salmon_index, 0))
        self.vh_env.step(action=(ActionEnum.WALK_TO_OBJECT, 0, object_index))
        tup = self.vh_env.step(action=(ActionEnum.PUTIN, salmon_index, object_index))
        self.vh_env.step(action=(ActionEnum.STOP, 0, 0))

        print(f'instruct sequence: {self.vh_env.print_instruct()}')
        self.assertTrue(float(tup[1]) < 0.)

    def test_f_env0_open_object_not_openable(self):
        self.vh_env.reset(options={
            'environment_index': 0,
        })

        food_index_dict = self.vh_env.get_food_index_dict()
        object_index_dict = self.vh_env.get_object_index_dict()
        object_index = object_index_dict['coffeetable']

        salmon_index = food_index_dict['salmon']
        pie_index = food_index_dict['pie']
        chocolatesyrup_index = food_index_dict['chocolatesyrup']

        self.vh_env.step(action=(ActionEnum.WALK_TO_FOOD, salmon_index, 0))
        self.vh_env.step(action=(ActionEnum.GRAB, salmon_index, 0))
        self.vh_env.step(action=(ActionEnum.WALK_TO_OBJECT, 0, object_index))
        tup = self.vh_env.step(action=(ActionEnum.OPEN, 0, object_index))
        self.vh_env.step(action=(ActionEnum.STOP, 0, 0))

        print(f'instruct sequence: {self.vh_env.print_instruct()}')
        self.assertTrue(float(tup[1]) < 0.)

    def test_f_env0_open_faraway_object(self):
        self.vh_env.reset(options={
            'environment_index': 0,
        })

        food_index_dict = self.vh_env.get_food_index_dict()
        object_index_dict = self.vh_env.get_object_index_dict()
        object_index = object_index_dict['fridge']

        salmon_index = food_index_dict['salmon']
        pie_index = food_index_dict['pie']
        chocolatesyrup_index = food_index_dict['chocolatesyrup']


        self.vh_env.step(action=(ActionEnum.WALK_TO_FOOD, salmon_index, 0))
        self.vh_env.step(action=(ActionEnum.GRAB, salmon_index, 0))
        # self.vh_env.step(action=(ActionEnum.WALK_TO_OBJECT, 0, object_index))
        tup = self.vh_env.step(action=(ActionEnum.OPEN, 0, object_index))
        self.vh_env.step(action=(ActionEnum.STOP, 0, 0))

        print(f'instruct sequence: {self.vh_env.print_instruct()}')
        self.assertTrue(float(tup[1]) < 0.)

    def test_f_env0_open_object_already_opened(self):
        self.vh_env.reset(options={
            'environment_index': 0,
        })

        food_index_dict = self.vh_env.get_food_index_dict()
        object_index_dict = self.vh_env.get_object_index_dict()
        object_index = object_index_dict['fridge']

        salmon_index = food_index_dict['salmon']
        pie_index = food_index_dict['pie']
        chocolatesyrup_index = food_index_dict['chocolatesyrup']


        self.vh_env.step(action=(ActionEnum.WALK_TO_FOOD, salmon_index, 0))
        self.vh_env.step(action=(ActionEnum.GRAB, salmon_index, 0))
        self.vh_env.step(action=(ActionEnum.WALK_TO_OBJECT, 0, object_index))
        self.vh_env.step(action=(ActionEnum.OPEN, 0, object_index))
        tup = self.vh_env.step(action=(ActionEnum.OPEN, 0, object_index))
        self.vh_env.step(action=(ActionEnum.STOP, 0, 0))

        print(f'instruct sequence: {self.vh_env.print_instruct()}')
        self.assertTrue(float(tup[1]) < 0.)

    def test_f_env0_close_object_not_openable(self):
        self.vh_env.reset(options={
            'environment_index': 0,
        })
        food_index_dict = self.vh_env.get_food_index_dict()
        object_index_dict = self.vh_env.get_object_index_dict()
        object_index = object_index_dict['coffeetable']

        salmon_index = food_index_dict['salmon']
        pie_index = food_index_dict['pie']
        chocolatesyrup_index = food_index_dict['chocolatesyrup']

        self.vh_env.step(action=(ActionEnum.WALK_TO_OBJECT, 0, object_index))
        self.vh_env.step(action=(ActionEnum.OPEN, 0, object_index))
        tup = self.vh_env.step(action=(ActionEnum.CLOSE, 0, object_index))
        self.vh_env.step(action=(ActionEnum.STOP, 0, 0))

        print(f'instruct sequence: {self.vh_env.print_instruct()}')
        self.assertTrue(float(tup[1]) < 0.)

    def test_f_env0_close_object_already_closed(self):
        self.vh_env.reset(options={
            'environment_index': 0,
        })

        food_index_dict = self.vh_env.get_food_index_dict()
        object_index_dict = self.vh_env.get_object_index_dict()
        object_index = object_index_dict['fridge']

        salmon_index = food_index_dict['salmon']
        pie_index = food_index_dict['pie']
        chocolatesyrup_index = food_index_dict['chocolatesyrup']

        self.vh_env.step(action=(ActionEnum.WALK_TO_OBJECT, 0, object_index))
        self.vh_env.step(action=(ActionEnum.OPEN, 0, object_index))
        self.vh_env.step(action=(ActionEnum.CLOSE, 0, object_index))
        tup = self.vh_env.step(action=(ActionEnum.CLOSE, 0, object_index))
        self.vh_env.step(action=(ActionEnum.STOP, 0, 0))

        print(f'instruct sequence: {self.vh_env.print_instruct()}')
        self.assertTrue(float(tup[1]) < 0.)

    def test_f_env0_close_faraway_object(self):
        self.vh_env.reset(options={
            'environment_index': 0,
        })
        food_index_dict = self.vh_env.get_food_index_dict()
        object_index_dict = self.vh_env.get_object_index_dict()
        object_index = object_index_dict['fridge']

        salmon_index = food_index_dict['salmon']
        pie_index = food_index_dict['pie']
        chocolatesyrup_index = food_index_dict['chocolatesyrup']

        self.vh_env.step(action=(ActionEnum.WALK_TO_OBJECT, 0, object_index))
        self.vh_env.step(action=(ActionEnum.OPEN, 0, object_index))
        self.vh_env.step(action=(ActionEnum.WALK_TO_FOOD, salmon_index, 0))
        self.vh_env.step(action=(ActionEnum.GRAB, salmon_index, 0))
        tup = self.vh_env.step(action=(ActionEnum.CLOSE, 0, object_index))
        self.vh_env.step(action=(ActionEnum.STOP, 0, 0))

        print(f'instruct sequence: {self.vh_env.print_instruct()}')
        self.assertTrue(float(tup[1]) < 0.)






if __name__ == '__main__':
    unittest.main()


