from src.vh_env import VirtualHomeGatherFoodEnvV2
from virtualhome.simulation.unity_simulator.comm_unity import UnityCommunication
import unittest

class VhGatherEnvTest(unittest.TestCase):
    # test virtual home gather food environment
    YOUR_FILE_NAME = "D:\\programs\\windows_exec.v2.2.4\\VirtualHome.exe"

    def setUp(self):
        self.comm = UnityCommunication(file_name=self.YOUR_FILE_NAME)

    def tearDown(self):
        self.comm.close()

    def test_t_place_food_one_by_one(self):

        self.comm.reset(0)
        self.comm.add_character()
        res, g = self.comm.environment_graph()
        vh_env = VirtualHomeGatherFoodEnvV2(environment_graph=g, log_level='debug')
        vh_env.reset()

        vh_env.step('open_fridge')
        vh_env.step('grab_food')
        vh_env.step('place_food')
        vh_env.step('close_fridge')

        vh_env.step('open_fridge')
        vh_env.step('grab_food')
        vh_env.step('place_food')
        vh_env.step('close_fridge')

        vh_env.step('open_fridge')
        vh_env.step('grab_food')
        vh_env.step('place_food')
        vh_env.step('close_fridge')

        tup = vh_env.step('end_game')
        print(f'instruction list: {vh_env.get_instruction_list()}')
        print(f'final state: {tup}')
        self.assertTrue(True)

    def test_t_place_food_one_by_one_2(self):
        self.comm.reset(0)
        self.comm.add_character()
        res, g = self.comm.environment_graph()
        vh_env = VirtualHomeGatherFoodEnvV2(environment_graph=g, log_level='debug')
        vh_env.reset()

        for i in range(3):
            vh_env.step('grab_food')
            vh_env.step('open_fridge')
            vh_env.step('place_food')
            vh_env.step('close_fridge')

        tup = vh_env.step('end_game')
        print(f'instruction list: {vh_env.get_instruction_list()}')
        print(f'final state: {tup}')
        self.assertTrue(True)

    def test_t_place_food_fullhands(self):
        self.comm.reset(0)
        self.comm.add_character()
        res, g = self.comm.environment_graph()
        vh_env = VirtualHomeGatherFoodEnvV2(environment_graph=g, log_level='debug')
        vh_env.reset()

        vh_env.step('grab_food')
        vh_env.step('open_fridge')
        vh_env.step('place_food')
        vh_env.step('grab_food')
        vh_env.step('grab_food')
        vh_env.step('place_food')
        vh_env.step('place_food')
        vh_env.step('close_fridge')

        tup = vh_env.step('end_game')
        print(f'instruction list: {vh_env.get_instruction_list()}')
        print(f'final state: {tup}')
        self.assertTrue(True)

    def test_f_grab_2plus_food(self):
        self.comm.reset(0)
        self.comm.add_character()
        res, g = self.comm.environment_graph()
        vh_env = VirtualHomeGatherFoodEnvV2(environment_graph=g, log_level='debug')
        vh_env.reset()

        vh_env.step('grab_food')
        vh_env.step('grab_food')
        tup = vh_env.step('grab_food')
        vh_env.step('end_game')

        print(f'instruction list: {vh_env.get_instruction_list()}')
        print(f'final state: {tup}')
        self.assertTrue(float(tup[1]) < 0.)

    def test_f_open_fridge_no_free_hands(self):
        self.comm.reset(0)
        self.comm.add_character()
        res, g = self.comm.environment_graph()
        vh_env = VirtualHomeGatherFoodEnvV2(environment_graph=g, log_level='debug')
        vh_env.reset()
        vh_env.step('grab_food')
        vh_env.step('grab_food')
        tup = vh_env.step('open_fridge')
        vh_env.step('end_game')

        print(f'instruction list: {vh_env.get_instruction_list()}')
        print(f'final state: {tup}')
        self.assertTrue(float(tup[1]) < 0.)

    def test_f_close_fridge_no_free_hands(self):
        self.comm.reset(0)
        self.comm.add_character()
        res, g = self.comm.environment_graph()
        vh_env = VirtualHomeGatherFoodEnvV2(environment_graph=g, log_level='debug')
        vh_env.reset()

        vh_env.step('open_fridge')
        vh_env.step('grab_food')
        vh_env.step('grab_food')
        tup = vh_env.step('close_fridge')
        vh_env.step('end_game')

        print(f'instruction list: {vh_env.get_instruction_list()}')
        print(f'final state: {tup}')
        self.assertTrue(float(tup[1]) < 0.)

    def test_f_put_food_to_close_fridge(self):
        self.comm.reset(0)
        self.comm.add_character()
        res, g = self.comm.environment_graph()
        vh_env = VirtualHomeGatherFoodEnvV2(environment_graph=g, log_level='debug')
        vh_env.reset()


        vh_env.step('grab_food')
        tup = vh_env.step('place_food')
        vh_env.step('end_game')

        print(f'instruction list: {vh_env.get_instruction_list()}')
        print(f'final state: {tup}')
        self.assertTrue(float(tup[1]) < 0.)

    def test_f_place_food_without_grabbed(self):
        self.comm.reset(0)
        self.comm.add_character()
        res, g = self.comm.environment_graph()

        vh_env = VirtualHomeGatherFoodEnvV2(environment_graph=g, log_level='debug')
        vh_env.reset()


        vh_env.step('open_fridge')
        tup = vh_env.step('place_food')
        vh_env.step('grab_food')
        vh_env.step('place_food')
        vh_env.step('end_game')

        print(f'instruction list: {vh_env.get_instruction_list()}')
        print(f'final state: {tup}')
        self.assertTrue(float(tup[1]) < 0.)

    def test_f_open_fridge_twice(self):
        self.comm.reset(0)
        self.comm.add_character()
        res, g = self.comm.environment_graph()

        vh_env = VirtualHomeGatherFoodEnvV2(environment_graph=g, log_level='debug')
        vh_env.reset()

        vh_env.step('open_fridge')
        tup = vh_env.step('open_fridge')
        vh_env.step('end_game')

        print(f'instruction list: {vh_env.get_instruction_list()}')
        print(f'final state: {tup}')
        self.assertTrue(float(tup[1]) < 0.)

    def test_f_close_fridge_twice(self):
        self.comm.reset(0)
        self.comm.add_character()
        res, g = self.comm.environment_graph()

        vh_env = VirtualHomeGatherFoodEnvV2(environment_graph=g, log_level='debug')
        vh_env.reset()

        # vh_env.step('open_fridge')
        vh_env.step(0)
        vh_env.step('grab_food')
        vh_env.step('grab_food')
        vh_env.step('place_food')
        vh_env.step('close_fridge')
        tup = vh_env.step('close_fridge')

        vh_env.step('end_game')

        print(f'instruction list: {vh_env.get_instruction_list()}')
        print(f'final state: {tup}')
        self.assertTrue(float(tup[1]) < 0.)

if __name__ == '__main__':
    unittest.main()
