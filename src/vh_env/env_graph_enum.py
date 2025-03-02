from enum import IntEnum

class ActionEnum(IntEnum):
    NONE = 0
    STOP = 1
    WALK_TO_FOOD = 2
    WALK_TO_OBJECT = 3
    GRAB = 4
    PUT = 5
    PUTIN = 6
    OPEN = 7
    CLOSE = 8


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

class FoodStateV2Enum(IntEnum):
    NONE = 0
    INITIAL = 1
    HOLD = 2
    PLACED = 3

class CharacterPlaceV2Enum(IntEnum):
    NONE = 0
    FOOD = 1
    FRIDGE = 2


if __name__ == '__main__':
    pass
