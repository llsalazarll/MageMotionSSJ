import vgamepad as vg


class GamepadController:

    def __init__(self):

        self.gamepad = vg.VX360Gamepad()

    def update_controls(
            self,
            steering,
            throttle,
            brake
    ):

        """
        steering: -1 to 1
        throttle: 0 to 1
        brake: 0 to 1
        """

        # Steering
        self.gamepad.left_joystick_float(
            x_value_float=steering,
            y_value_float=0.0
        )

        # Throttle
        self.gamepad.right_trigger_float(
            value_float=throttle
        )

        # Brake
        self.gamepad.left_trigger_float(
            value_float=brake
        )

        self.gamepad.update()