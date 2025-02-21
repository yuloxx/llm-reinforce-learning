from enum import IntEnum

class ActionEnum(IntEnum):
    NONE = 0
    WALK_TO_FOOD = 1
    WALK_TO_OBJECT = 2
    GRAB = 3
    PUT = 4
    PUTIN = 5
    OPEN = 6
    CLOSE = 7

# BitMap
class FoodStateBitmapEnum(IntEnum):
    NONE = 0
    EXIST = 1

# BitMap
class ObjectStateBitmapEnum(IntEnum):
    NONE = 0
    EXIST = 1
    OPEN = 2
    CAN_OPEN = 4
    TURNED_ON = 8
    HAS_SWITCH = 16


class FoodCharacterStateEnum(IntEnum):
    NONE = 0
    CLOSE_TO = 1
    HOLD = 2

class ObjectCharacterStateEnum(IntEnum):
    NONE = 0
    CLOSE_TO = 1
    FACING = 2

class FoodObjectStateEnum(IntEnum):
    NONE = 0
    ON = 1
    INSIDE = 2
    FACING = 3


if __name__ == '__main__':
    pass
