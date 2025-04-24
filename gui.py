import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from tkinter import messagebox
import numpy as np
from numpy import deg2rad as rad
from numpy import rad2deg as deg
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import matplotlib.animation as animation
import quaternionic
from quaternionic import converters
from dual_quaternions import DualQuaternion
import time

from helper_functions import *
from bspline import *

import pymycobot
from packaging import version

if version.parse(pymycobot.__version__) >= version.parse("3.6.0"):
    from pymycobot import MyCobot280 as MyCobot
else:
    from pymycobot import MyCobot

from pymycobot import PI_PORT, PI_BAUD
import roboticstoolbox as rtb
import os
import csv

dual_quaternions = []
path_coords = []
selected_coords = []
Spline_degree = 1
control_points = None
move_speed = 30
urdf_path = os.getcwd() + "/mycobot_280_pi.urdf"
arm = rtb.Robot.URDF(urdf_path)

ani = None  # Animation object
animated_marker = None  # Animation marker handle
fig = None
ax = None
mc = None


# --- Animation Functions ---
def init_animation():
    """Initializes or clears the animation marker."""
    global animated_marker, ax  # Corrected: Added global ax
    if ax is None:
        return  # Guard against calling before ax is initialized

    # Remove previous marker if it exists
    if animated_marker:
        try:
            if animated_marker in ax.lines:
                ax.lines.remove(animated_marker)
            elif animated_marker in ax.collections:
                ax.collections.remove(animated_marker)
        except (ValueError, AttributeError):
            pass  # Already removed or None
        animated_marker = None

    # Create a new invisible marker (plot used for 3D updates)
    (animated_marker,) = ax.plot(
        [], [], [], marker="o", color="purple", markersize=8, linestyle=""
    )
    return (animated_marker,)


def animate_path(i, path_data):
    """Updates the marker position for each animation frame."""
    global animated_marker  # Access the marker created in init_animation
    if i < len(path_data) and animated_marker:
        pose = path_data[i]
        # Coordinates are expected in mm, convert to m for plotting
        x, y, z = pose[0] / 1000, pose[1] / 1000, pose[2] / 1000
        animated_marker.set_data_3d([x], [y], [z])  # Update marker position
    return (animated_marker,)


# --- GUI Functions ---
def reset():
    """Resets the GUI state, including the animation."""
    global dual_quaternions, path_coords, selected_coords, Spline_degree, control_points, ani, animated_marker

    # Stop any existing animation
    if ani is not None:
        try:
            ani.event_source.stop()
        except AttributeError:
            pass
        ani = None

    # Clear the animation marker
    init_animation()  # This will remove the old marker and prepare a new (invisible) one

    # Reset other variables
    dual_quaternions = []
    path_coords = []
    selected_coords = []
    Spline_degree = 1
    control_points = None
    if "listbox" in globals() and listbox:  # Check if listbox exists
        listbox.delete(0, tk.END)
    if "degree_textbox" in globals() and degree_textbox:
        degree_textbox.delete(0, tk.END)
        degree_textbox.insert(0, "1")  # Reset to default
    if "control_pts_textbox" in globals() and control_pts_textbox:
        control_pts_textbox.delete(0, tk.END)
    update_motion()  # Update plot (will be cleared)


def open_file():
    """Opens a file, parses poses, and updates the display."""
    reset()  # Reset everything before loading new data
    global selected_coords
    file = filedialog.askopenfilename(
        title="Select File", filetypes=(("Text files", "*.txt*"), ("all files", "*.*"))
    )
    if file:
        try:
            selected_coords = parse_pose(file)
            if not selected_coords:  # Handle empty file or parse error
                messagebox.showerror(
                    "File Error",
                    "Could not parse coordinates from file or file is empty.",
                )
                return
            print(f"Loaded {len(selected_coords)} coordinates.")  # Debug print
            update_motion()  # Update display with new data
        except Exception as e:
            messagebox.showerror("File Error", f"Failed to open or parse file:\n{e}")
            print(f"File Error: {e}")  # Debug print
            reset()


def save_to_file():
    """Saves the current control positions to a file."""
    if not selected_coords:
        messagebox.showwarning("Save Error", "No control positions to save.")
        return
    file_path = filedialog.asksaveasfilename(
        defaultextension=".txt",
        filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
    )
    if file_path:
        try:
            with open(file_path, mode="w", newline="") as file:
                writer = csv.writer(file, delimiter=",")
                for row in selected_coords:
                    writer.writerow(row)
            messagebox.showinfo("Save Successful", f"Positions saved to {file_path}")
        except Exception as e:
            messagebox.showerror("Save Error", f"Failed to save file:\n{e}")


def set_coords():
    """Opens a window to collect control poses by manually moving the robot."""
    global selected_coords, mc  # Ensure mc is accessible

    # Check if robot communication is active
    if mc is None:
        messagebox.showerror(
            "Robot Error",
            "Robot communication object 'mc' not initialized. Check connection settings and uncomment the line 'mc = MyCobot(...)' if using the robot.",
        )
        return
    try:
        # A more reliable check might be needed depending on the library version
        mc.get_angles()  # Try a simple command
        print("Robot connection confirmed for collecting poses.")
    except Exception as e:
        messagebox.showerror("Robot Error", f"Robot communication error:\n{e}")
        return

    new_selected_coords_temp = []  # Temporary list for the popup

    win = tk.Toplevel(root)
    win.wm_title("Collect Control Poses")
    win.geometry("400x400")
    win.grid_columnconfigure(0, weight=1)
    win.grid_columnconfigure(1, weight=1)
    win.grab_set()  # Make window modal
    win.focus_set()

    # --- Popup Window Widgets ---
    popup_listbox = tk.Listbox(
        win,
        height=15,
        width=50,
        bg="light grey",
        activestyle="dotbox",
        font=("Helvetica", 8),
    )
    popup_listbox.grid(row=1, column=0, columnspan=2, padx=10, pady=10, sticky="nsew")
    win.grid_rowconfigure(1, weight=1)  # Make listbox expandable

    scrollbar_popup = ttk.Scrollbar(
        win, orient=tk.VERTICAL, command=popup_listbox.yview
    )
    scrollbar_popup.grid(row=1, column=2, sticky="ns")
    popup_listbox["yscrollcommand"] = scrollbar_popup.set

    def update_popup_listbox():
        popup_listbox.delete(0, tk.END)
        for i, c in enumerate(new_selected_coords_temp):
            # Format for display
            formatted_coords = ", ".join(f"{x:.2f}" for x in c)
            popup_listbox.insert(tk.END, f"{i}: [{formatted_coords}]")

    def add_current_position():
        try:
            current_coords = mc.get_coords()
            if current_coords:
                # Ensure coords are floats and rounded (optional)
                current_coords = [float(f"{x:.2f}") for x in current_coords]
                new_selected_coords_temp.append(current_coords)
                update_popup_listbox()
            else:
                messagebox.showwarning(
                    "Robot Warning",
                    "Could not retrieve robot coordinates (received empty list).",
                )
        except Exception as e:
            messagebox.showerror("Robot Error", f"Error getting coordinates:\n{e}")

    def save_and_close():
        global selected_coords
        reset()  # Reset main GUI before applying new coords
        selected_coords = new_selected_coords_temp[:]  # Copy collected coords
        update_motion()  # Update main GUI
        win.destroy()

    # Buttons
    button_frame_popup = tk.Frame(win)
    button_frame_popup.grid(row=0, column=0, columnspan=2, pady=5)

    add_button = ttk.Button(
        button_frame_popup, text="Add Current Position", command=add_current_position
    )
    add_button.pack(side=tk.LEFT, padx=5)

    save_button = ttk.Button(
        button_frame_popup, text="Save and Close", command=save_and_close
    )
    save_button.pack(side=tk.LEFT, padx=5)

    # Initial update
    update_popup_listbox()


def update_listbox():
    """Updates the main listbox with current selected_coords."""
    if "listbox" not in globals() or not listbox:
        return  # Check existence
    listbox.delete(0, tk.END)
    if selected_coords:
        for i, c in enumerate(selected_coords):
            # Format for display
            try:
                formatted_coords = ", ".join(f"{x:.2f}" for x in c)
                listbox.insert(tk.END, f"{i}: [{formatted_coords}]")
            except (
                TypeError
            ):  # Handle case where c might not be iterable or contain non-numerics
                listbox.insert(tk.END, f"{i}: [Error formatting data]")


def update_motion():
    """Clears plot, redraws axes and control points, and recalculates path."""
    global dual_quaternions, ani, ax, fig, canvas  # Ensure ax, fig, canvas are accessible

    if ax is None or fig is None or canvas is None:
        print("Plot axes not initialized yet. Skipping update.")
        return

    # Stop and clear any previous animation before redrawing
    if ani is not None:
        try:
            ani.event_source.stop()
        except AttributeError:
            pass
        ani = None
    init_animation()  # Ensure marker is cleared/reset

    # Clear the plot axes
    ax.clear()
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    # Set bounds every time, as clear() removes them
    ax.set_xbound(-0.4, 0.4)
    ax.set_ybound(-0.4, 0.4)
    ax.set_zlim(0, 0.6)
    # Draw a simple origin marker
    ax.scatter(0, 0, 0, marker="+", color="black")

    # Update the listbox display
    update_listbox()

    # Recalculate dual quaternions and draw control point axes
    dual_quaternions = []
    if selected_coords:
        print(
            f"Processing {len(selected_coords)} selected coordinates for plotting."
        )  # Debug
        for i, coord in enumerate(selected_coords):
            try:
                if len(coord) != 6:
                    print(f"Skipping invalid coordinate at index {i}: {coord}")
                    continue
                # Convert pose to dual quaternion
                q_rot = quaternionic.array(
                    to_quaternion(rad(coord[3]), rad(coord[4]), rad(coord[5]))
                )
                # Pose format for DualQuaternion: [w, x, y, z, x_trans, y_trans, z_trans] (translation in meters)
                pose_for_dq = q_rot.tolist() + [c / 1000 for c in coord[0:3]]
                temp_dq = DualQuaternion.from_quat_pose_array(pose_for_dq)
                dual_quaternions.append(temp_dq)
                # Draw axis for this control point (coords are in mm)
                draw_axis(coord, ax, np)
            except Exception as e:
                print(f"Error processing coordinate at index {i} ({coord}): {e}")
                # messagebox.showwarning("Coordinate Error", f"Could not process coordinate:\n{coord}\nError: {e}") # Can be too noisy
                continue  # Skip this point
        print(f"Calculated {len(dual_quaternions)} dual quaternions.")  # Debug

    # Attempt to create the path (which now includes starting the animation)
    if len(dual_quaternions) >= 2:  # Need at least 2 points for a path
        print("Attempting to create path...")  # Debug
        createPath()
    else:
        print(
            "Not enough valid dual quaternions to create path. Clearing path data."
        )  # Debug
        path_coords.clear()  # Ensure path is cleared if not enough points
        canvas.draw()  # Update canvas even if no path is drawn


def createPath():
    """Calculates the B-spline path and starts the animation."""
    global path_coords, dual_quaternions, joint_array, passed, ani, animated_marker, Spline_degree, control_points, ax, fig, canvas  # Make sure plot objects are global

    if ax is None or fig is None or canvas is None:
        print("Plot axes not initialized for path creation.")
        return

    # Stop previous animation
    if ani is not None:
        try:
            ani.event_source.stop()
        except AttributeError:
            pass
        ani = None
    init_animation()  # Reset marker

    passed = True  # Assume path is reachable initially
    path_coords = []  # Store calculated poses [x, y, z, r, p, y] in mm/deg
    joint_array = []  # Store corresponding joint angles
    points_resolution = 30  # Number of points to calculate along the spline

    # Need at least 2 valid dual quaternions
    if len(dual_quaternions) < 2:
        print("createPath: Not enough dual quaternions.")
        canvas.draw()  # Update plot even if empty
        return

    # Prepare control points based on curve type
    adjusted_control_pos_dq = dual_quaternions[:]  # Make a copy
    num_data_points = len(dual_quaternions)  # Number of input data points (DQ)

    # --- Degree and Control Point Validation ---
    try:
        current_degree = (
            int(degree_textbox.get()) if degree_textbox.get().isdigit() else 1
        )
        Spline_degree = (
            current_degree  # Update global Spline_degree based on current textbox value
        )
    except tk.TclError:  # Handle case where degree_textbox might not exist yet
        current_degree = 1
        Spline_degree = 1

    try:
        current_control_pts_input = (
            int(control_pts_textbox.get())
            if control_pts_textbox.get().isdigit()
            else None
        )
        # Assuming user inputs TOTAL control points (H). We need 'h' for bspline.py funcs?
        # Let's stick to the logic from check_control_pts, storing TOTAL in control_points global
        control_points = (
            current_control_pts_input  # Update global based on current textbox
        )
    except tk.TclError:  # Handle case where textbox might not exist
        current_control_pts_input = None
        control_points = None

    if selected_curve.get() == "B-spline Motion":
        # Check degree vs number of (potentially adjusted) control points
        num_motion_ctrl_pts = len(adjusted_control_pos_dq)
        if selected_type.get() == "Closed":
            if num_data_points > current_degree:
                adjusted_control_pos_dq.extend(dual_quaternions[0:current_degree])
                num_motion_ctrl_pts = len(adjusted_control_pos_dq)  # Update count
            else:
                print(
                    "Warning: Not enough control points for closed loop of degree",
                    current_degree,
                )
                # Proceeding, but knot vector generation might be problematic

        if current_degree >= num_motion_ctrl_pts:
            messagebox.showerror(
                "Input Error",
                f"Curve degree ({current_degree}) must be less than the number of control points ({num_motion_ctrl_pts}).",
            )
            passed = False

    elif selected_curve.get() == "B-spline Interpolation":
        # Validation based on TOTAL control points input (control_points global)
        if control_points is None:
            messagebox.showerror(
                "Input Error",
                "Number of Total Control Points must be specified for Interpolation.",
            )
            passed = False
        elif control_points < current_degree + 1:
            messagebox.showerror(
                "Input Error",
                f"Total Control Pts ({control_points}) must be >= Degree+1 ({current_degree + 1}).",
            )
            passed = False
        elif control_points > num_data_points:
            messagebox.showerror(
                "Input Error",
                f"Total Control Pts ({control_points}) must be <= Number of Data Points ({num_data_points}).",
            )
            passed = False

    # --- Path Calculation Logic ---
    if passed:  # Only proceed if initial validation passed
        try:
            if selected_curve.get() == "B-spline Motion":
                print(
                    f"Calculating B-spline Motion: Degree={current_degree}, CtrlPts={len(adjusted_control_pos_dq)}, End={selected_type.get()}"
                )
                knot_vector = gen_knot_vector(
                    degree=current_degree,
                    n=len(adjusted_control_pos_dq) - 1,
                    style=selected_type.get(),
                )
                print(f"Knot Vector: {knot_vector}")  # Debug
                b_spline_dqs = b_spline_curve(
                    knot_vector=knot_vector,
                    degree=current_degree,
                    control_positions=adjusted_control_pos_dq,
                    resolution=points_resolution,
                )

                for i, dq in enumerate(b_spline_dqs):
                    temp_quat_pose = dq.quat_pose_array()
                    temp_matrix = dq.homogeneous_matrix()

                    # Inverse Kinematics Check - Corrected: Removed 'seed' argument
                    ik_solution = arm.ets().ik_GN(
                        temp_matrix, ilimit=50, slimit=300, joint_limits=True
                    )
                    joint_array.append(ik_solution)

                    if not ik_solution[
                        1
                    ]:  # Check if IK succeeded (ik_solution[1] is success flag)
                        print(
                            f"IK Failed at point {i}: Target Matrix:\n{temp_matrix}\nResult: {ik_solution}"
                        )
                        if passed:  # Show error only once
                            messagebox.showerror(
                                title="Robot Workspace Check",
                                message=f"Manipulator unable to reach configuration near point {i}.",
                            )
                        passed = False
                        path_coords = []  # Clear partial path
                        break  # Stop path generation on first failure

                    temp_angles = to_euler_angles(
                        temp_quat_pose[0:4]
                    )  # Get Euler angles in degrees
                    temp_pose_translation = [
                        c * 1000 for c in get_translation(dq)
                    ]  # Get translation in mm
                    temp_pose = (
                        temp_pose_translation + temp_angles
                    )  # Combine [x,y,z, r,p,y]
                    path_coords.append(temp_pose)  # Append the pose

                    # Draw axes along the path occasionally for static visualization
                    if i % 5 == 0:
                        draw_axis(temp_pose, ax, np)  # Draw full axis

            elif selected_curve.get() == "B-spline Interpolation":
                print(
                    f"Calculating B-spline Interpolation: Degree={current_degree}, DataPts={num_data_points}, TotalCtrlPts={control_points}"
                )
                parameter = parameterize(
                    dual_quaternions, selected_parameter.get(), selected_distance.get()
                )

                # Calculate 'h' for get_control_points/interpolation_knot_vector
                # If control_points is TOTAL (H), then h = H - 1 (as per Piegl's notation n+1 points, h+1 ctrl pts, m+1 knots)
                h_internal_count = control_points - 1  # h = H - 1

                # Call get_control_points (assuming it expects 'h' as internal count or similar)
                # The bspline.py get_control_points seems complex, need to verify its 'h' parameter meaning.
                # Let's assume h = total_ctrl_pts - 1 (as used in Piegl)
                print(f"Calling get_control_points with h={h_internal_count}")  # Debug
                control_points_arr = get_control_points(
                    dual_quaternions, parameter, current_degree, h=h_internal_count
                )
                print(
                    f"Calculated {len(control_points_arr)} internal control points."
                )  # Debug

                # n = num_data_points - 1
                n_data_index = num_data_points - 1
                knot_vector = interpolation_knot_vector(
                    n=n_data_index,
                    h=h_internal_count,
                    degree=current_degree,
                    parameter=parameter,
                )
                print(f"Interpolation Knot Vector: {knot_vector}")  # Debug

                b_spline_dqs = b_spline_curve(
                    knot_vector=knot_vector,
                    degree=current_degree,
                    control_positions=control_points_arr,
                    resolution=points_resolution,
                )

                for i, dq in enumerate(b_spline_dqs):
                    temp_quat_pose = dq.quat_pose_array()
                    temp_matrix = dq.homogeneous_matrix()
                    # Inverse Kinematics Check - Corrected: Removed 'seed' argument
                    ik_solution = arm.ets().ik_GN(
                        temp_matrix, ilimit=50, slimit=300, joint_limits=True
                    )
                    joint_array.append(ik_solution)

                    if not ik_solution[1]:
                        print(f"IK Failed at point {i} (Interpolation)")
                        if passed:
                            messagebox.showerror(
                                title="Robot Workspace Check",
                                message=f"Manipulator unable to reach configuration near point {i} (Interpolation).",
                            )
                        passed = False
                        path_coords = []
                        break

                    temp_angles = to_euler_angles(temp_quat_pose[0:4])
                    temp_pose_translation = [c * 1000 for c in get_translation(dq)]
                    temp_pose = temp_pose_translation + temp_angles
                    path_coords.append(temp_pose)

                    if i % 5 == 0:
                        draw_axis(temp_pose, ax, np)

        except Exception as e:
            messagebox.showerror(
                "Path Calculation Error",
                f"An error occurred during path calculation:\n{e}",
            )
            print(f"Path calculation error: {e}")  # Print detailed error
            passed = False
            path_coords = []  # Clear any partial path

    # --- Start Animation ---
    if path_coords and passed:
        print(
            f"Path calculated successfully with {len(path_coords)} points. Starting animation."
        )
        # Plot the static path line
        path_xyz = np.array(
            [(p[0] / 1000, p[1] / 1000, p[2] / 1000) for p in path_coords]
        )
        ax.plot(
            path_xyz[:, 0], path_xyz[:, 1], path_xyz[:, 2], color="cyan", linestyle="--"
        )

        total_duration_ms = 3000
        frame_interval = max(10, int(total_duration_ms / len(path_coords)))

        ani = animation.FuncAnimation(
            fig,
            animate_path,
            frames=len(path_coords),
            init_func=init_animation,
            fargs=(path_coords,),
            interval=frame_interval,
            blit=False,
            repeat=False,
        )
    elif not passed:
        print("Path calculation failed or path is unreachable. No animation.")
        init_animation()  # Ensure marker is cleared
    else:
        print("Not enough points or no valid path generated. No animation.")
        init_animation()  # Ensure marker is cleared

    canvas.draw()  # Update the plot display


def find_distance(point1, point2):
    """Calculates Euclidean distance between two 3D points (first 3 elements)."""
    # Add checks for valid input points
    if not point1 or not point2 or len(point1) < 3 or len(point2) < 3:
        return 0.0
    try:
        return (
            (point1[0] - point2[0]) ** 2
            + (point1[1] - point2[1]) ** 2
            + (point1[2] - point2[2]) ** 2
        ) ** 0.5
    except (TypeError, IndexError):
        return 0.0


def run_motion():
    """Sends the calculated path coordinates to the MyCobot arm."""
    global passed, mc  # Ensure mc is accessible

    if not path_coords:
        messagebox.showwarning("Motion Warning", "No path calculated to run.")
        return
    if not passed:
        messagebox.showerror(
            "Motion Error", "Calculated path is unreachable. Cannot run motion."
        )
        return

    # Check robot connection
    if mc is None:
        messagebox.showerror(
            "Robot Error",
            "Robot communication object 'mc' not initialized. Cannot run motion.",
        )
        return
    try:
        mc.get_angles()  # Try a simple command
        print("Robot connection confirmed for running motion.")
    except Exception as e:
        messagebox.showerror("Robot Error", f"Robot communication error:\n{e}")
        return

    # Send coordinates one by one
    try:
        print("Starting robot motion...")
        # Go to the first point
        if not path_coords:
            return  # Should not happen if checked earlier, but safety first
        print(f"Sending point 0: {path_coords[0]} Speed: {move_speed}")
        mc.send_coords(
            path_coords[0], speed=move_speed, mode=1
        )  # mode=1 for linear motion

        # Wait logic: Use time.sleep for a duration based on estimated move time,
        # as is_moving() might not be reliable. Calculate time based on distance?
        # Simple fixed delay for now after the first point.
        time.sleep(1.5)  # Adjust as needed based on typical first move time

        # Send remaining points
        for i, c in enumerate(path_coords[1:], 1):
            print(f"Sending point {i}: {c} Speed: {move_speed}")
            mc.send_coords(c, speed=move_speed, mode=1)
            # Calculate estimated time based on distance to previous point
            dist = find_distance(c, path_coords[i - 1])
            estimated_time = (
                (dist / move_speed) if move_speed > 0 else 0.5
            )  # Avoid division by zero
            wait_time = max(0.2, estimated_time * 1.1)  # Add a buffer, min wait 0.2s
            print(
                f"  Distance: {dist:.2f} mm, Est. Time: {estimated_time:.2f} s, Waiting: {wait_time:.2f} s"
            )
            time.sleep(wait_time)

        print("Robot motion finished (commands sent).")
        messagebox.showinfo(
            "Motion Complete", "Robot has finished executing the path (commands sent)."
        )

    except Exception as e:
        messagebox.showerror(
            "Robot Motion Error", f"An error occurred during motion execution:\n{e}"
        )
        print(f"Robot motion error: {e}")


def release_servo():
    """Releases all servos on the MyCobot."""
    global mc
    if mc is None:
        messagebox.showerror(
            "Robot Error", "Robot communication object 'mc' not initialized."
        )
        return
    try:
        mc.release_all_servos()
        print("Servos released.")
        messagebox.showinfo("Servos", "All servos have been released.")
    except Exception as e:
        messagebox.showerror("Robot Error", f"Failed to release servos:\n{e}")


# --- Validation and Callback Functions ---
def checkDegree(value):
    """Validation function for the degree Entry widget."""
    global Spline_degree
    if value.isdigit():
        num_points = len(selected_coords)  # Use current selected_coords count
        # Basic check: degree >= 1
        if int(value) < 1:
            print("Degree validation failed: Must be >= 1")
            return False
        # Degree must be < number of control points (in B-spline Motion case)
        # For interpolation, Degree < Total Control Points (H)
        # Let's keep it simple: just validate it's a digit >= 1
        Spline_degree = int(value)
        print(f"Degree set to: {Spline_degree}")
        # No update_motion here, let createPath read the value
        return True
    elif value == "":
        Spline_degree = 1  # Default if empty
        print("Degree field empty, defaulting to 1")
        return True
    else:
        print(f"Degree validation failed: Non-digit '{value}'")
        return False  # Not a digit


def check_control_pts(value):
    """Validation function for the number of control points Entry (Interpolation)."""
    global control_points
    if value.isdigit():
        num_data_points = len(selected_coords)
        input_val = int(value)
        # Assuming user inputs TOTAL control points (H)
        # Check: H >= 2 (minimum for a line), H <= num_data_points
        if input_val >= 2 and input_val <= num_data_points:
            control_points = input_val  # Store total count
            print(f"Total Control Points (Interpolation) set to: {control_points}")
            # No update_motion here
            return True
        else:
            print(
                f"Total Control points validation failed: {input_val} vs data points {num_data_points}"
            )
            return False
    elif value == "":
        control_points = None  # Reset if empty
        print("Control points field empty.")
        return True
    else:
        print(f"Control points validation failed: Non-digit '{value}'")
        return False


def change_curve(*args):  # Use *args to accept Tkinter callback arguments
    """Callback function when a dropdown selection changes."""
    # Corrected: Removed the problematic print statement that caused NameErrors
    # print(f"Dropdown changed: Curve={selected_curve.get()}, Type={selected_type.get()}, Param={selected_parameter.get()}, Dist={selected_distance.get()}")
    print("Dropdown changed. Triggering update_motion.")  # Simple debug print
    update_motion()  # Recalculate path when settings change


def go_to_positions():
    """Commands the robot to move directly to each selected control position."""
    global mc
    if not selected_coords:
        messagebox.showwarning("Motion Warning", "No positions selected to move to.")
        return

    # Check robot connection
    if mc is None:
        messagebox.showerror(
            "Robot Error", "Robot communication object 'mc' not initialized."
        )
        return
    try:
        mc.get_angles()  # Check connection
        print("Robot connection confirmed for Go To Positions.")
    except Exception as e:
        messagebox.showerror("Robot Error", f"Robot communication error:\n{e}")
        return

    print("Moving robot to selected positions sequentially...")
    reachable_coords_to_move = []
    all_positions_checked = True

    # Check reachability for all points first
    for i, c in enumerate(selected_coords):
        try:
            if len(c) != 6:
                continue  # Skip invalid format
            q_rot = quaternionic.array(to_quaternion(rad(c[3]), rad(c[4]), rad(c[5])))
            pose_for_dq = q_rot.tolist() + [x / 1000 for x in c[0:3]]
            temp_dq = DualQuaternion.from_quat_pose_array(pose_for_dq)
            target_matrix = temp_dq.homogeneous_matrix()
            ik_sol = arm.ets().ik_GN(
                target_matrix, ilimit=30, slimit=100, joint_limits=True
            )
            if ik_sol[1]:  # If reachable
                reachable_coords_to_move.append(c)
            else:
                all_positions_checked = False
                if not messagebox.askyesno(
                    "Reachability Warning",
                    f"Position {i} [{c[0]:.1f}, ...] might be unreachable. Continue checking and attempt motion?",
                ):
                    print(
                        "Motion cancelled by user due to potential reachability issue."
                    )
                    return  # Stop checking and motion
                else:
                    # User chose to continue, we might still try to move there later
                    reachable_coords_to_move.append(
                        c
                    )  # Add anyway, but user was warned
        except Exception as e:
            all_positions_checked = False
            if not messagebox.askyesno(
                "IK Error",
                f"Error checking reachability for position {i}:\n{e}\nContinue checking and attempt motion?",
            ):
                print("Motion cancelled by user due to IK error.")
                return
            else:
                reachable_coords_to_move.append(c)  # Add anyway

    # Move to the points (potentially including ones user was warned about)
    try:
        for i, c in enumerate(reachable_coords_to_move):
            print(f"Sending robot to position {i}: {c}")
            mc.send_coords(coords=c, speed=20, mode=1)  # Use speed 20
            # Wait for move completion - Simple sleep for now
            # Ideally, calculate time based on distance from previous known position
            time.sleep(2.0)  # Fixed delay, adjust as needed
            print(f"Assumed reached position {i}.")
        print("Finished moving to all target positions.")
        messagebox.showinfo(
            "Motion Complete", "Robot finished moving to selected positions."
        )
    except Exception as e:
        messagebox.showerror(
            "Robot Motion Error", f"Error during 'Go To Positions':\n{e}"
        )
        print(f"Error during 'Go To Positions': {e}")


# --- GUI Setup ---
root = tk.Tk()
root.wm_title("B-Spline Motion Interface")
root.geometry("1000x700")  # Adjusted size slightly

# --- MyCobot Initialization ---
# Encapsulate in try-except
try:
    # === CHOOSE ONE CONNECTION METHOD ===
    # mc = MyCobot(PI_PORT, PI_BAUD) # For Raspberry Pi GPIO connection
    mc = MyCobot("/dev/ttyJETCOBOT", 1000000)
    # ====================================
    print("Attempting to connect to MyCobot...")
    # Optional: Wake up robot or check status
    # mc.set_fresh_mode(1) # May not be needed or available on all firmwares
    # time.sleep(0.5)
    if mc.is_controller_connected():
        print("Robot Connected Successfully.")
    else:
        print("Robot connection check failed (is_controller_connected returned False).")
        mc = None  # Ensure mc is None if check fails
        messagebox.showwarning(
            "Robot Connection",
            "Could not verify connection (is_controller_connected failed).\n\nRobot control features might be unstable.",
        )
except Exception as e:
    print(f"Failed to initialize MyCobot: {e}")
    messagebox.showwarning(
        "Robot Connection",
        f"Could not connect to MyCobot.\n{e}\n\nRobot control features will be disabled.",
    )
    mc = None  # Ensure mc is None if connection fails


# --- GUI Layout ---
# Left Frame (Controls)
left_frame = tk.Frame(root, width=450)
left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)
left_frame.pack_propagate(False)  # Prevent frame from shrinking

# Listbox Frame
listbox_frame = ttk.LabelFrame(
    left_frame, text="Control Positions (x,y,z mm; r,p,y deg)", padding=(5, 5)
)
listbox_frame.pack(fill=tk.X, pady=5)

listbox_inner_frame = tk.Frame(listbox_frame)
listbox_inner_frame.pack(
    fill=tk.BOTH, expand=True, side=tk.LEFT, padx=(0, 5)
)  # Pack listbox frame first

listbox = tk.Listbox(
    listbox_inner_frame,
    height=10,
    width=45,
    bg="light grey",
    activestyle="dotbox",
    font=("Helvetica", 9),
)  # Adjusted width slightly
listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

scrollbar = ttk.Scrollbar(
    listbox_inner_frame, orient=tk.VERTICAL, command=listbox.yview
)
scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
listbox["yscrollcommand"] = scrollbar.set

# Upper Button Frame (Save, Go To) - Placed next to the listbox
upper_button_frame = tk.Frame(listbox_frame)
upper_button_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(5, 0))
saveButton = ttk.Button(upper_button_frame, text="Save Positions", command=save_to_file)
saveButton.pack(pady=(0, 5), fill=tk.X)
go_to_position = ttk.Button(
    upper_button_frame, text="Go To Positions", command=go_to_positions
)
go_to_position.pack(pady=(0, 5), fill=tk.X)


# Input Frame (Settings)
input_frame = ttk.LabelFrame(left_frame, text="Motion Settings", padding=(10, 5))
input_frame.pack(fill=tk.X, pady=10)
input_frame.grid_columnconfigure(
    1, weight=1
)  # Allow dropdowns/entries to expand slightly

# Curve Type
curve = ["B-spline Motion", "B-spline Interpolation"]
selected_curve = tk.StringVar(value="B-spline Motion")
selected_curve.trace_add("write", change_curve)
curve_label = ttk.Label(input_frame, text="Curve Generation:")
curve_label.grid(row=0, column=0, sticky="w", padx=5, pady=3)
curve_dropdown = ttk.OptionMenu(
    input_frame, selected_curve, selected_curve.get(), *curve
)
curve_dropdown.grid(
    row=0, column=1, columnspan=2, sticky="ew", padx=5, pady=3
)  # Span 2 columns

# Degree
deg_label = ttk.Label(input_frame, text="Curve Degree:")
deg_label.grid(row=1, column=0, sticky="w", padx=5, pady=3)
deg_valid = root.register(checkDegree)
degree_textbox = ttk.Entry(
    input_frame, width=10, validate="key", validatecommand=(deg_valid, "%P")
)
degree_textbox.grid(row=1, column=1, sticky="w", padx=5, pady=3)
degree_textbox.insert(0, "1")  # Default value
deg_tooltip = ttk.Label(
    input_frame, text="(≥ 1)", font=("Helvetica", 8), foreground="grey"
)
deg_tooltip.grid(row=1, column=2, sticky="w", padx=5)


# Ending Condition (for B-spline Motion)
types = ["Clamped", "Closed"]
type_label = ttk.Label(input_frame, text="End Condition:")
type_label.grid(row=2, column=0, sticky="w", padx=5, pady=3)
selected_type = tk.StringVar(value="Clamped")
selected_type.trace_add("write", change_curve)
type_dropdown = ttk.OptionMenu(input_frame, selected_type, selected_type.get(), *types)
type_dropdown.grid(
    row=2, column=1, columnspan=2, sticky="ew", padx=5, pady=3
)  # Span 2 columns

# Interpolation Settings Frame
interp_frame = ttk.LabelFrame(
    left_frame, text="Interpolation Specific", padding=(10, 5)
)
interp_frame.pack(fill=tk.X, pady=5)
interp_frame.grid_columnconfigure(1, weight=1)

# Parameterization (Interpolation)
parameters = ["Uniform", "Chord", "Centripetal"]
parameter_label = ttk.Label(interp_frame, text="Parameterization:")
parameter_label.grid(row=0, column=0, sticky="w", padx=5, pady=3)
selected_parameter = tk.StringVar(value="Uniform")
selected_parameter.trace_add("write", change_curve)
parameter_dropdown = ttk.OptionMenu(
    interp_frame, selected_parameter, selected_parameter.get(), *parameters
)
parameter_dropdown.grid(
    row=0, column=1, columnspan=2, sticky="ew", padx=5, pady=3
)  # Span 2 columns

# Distance Metric (Interpolation - Chord/Centripetal)
distance = ["Quaternion", "Cartesian"]
distance_label = ttk.Label(interp_frame, text="Distance Metric:")
distance_label.grid(row=1, column=0, sticky="w", padx=5, pady=3)
selected_distance = tk.StringVar(value="Quaternion")
selected_distance.trace_add("write", change_curve)
distance_dropdown = ttk.OptionMenu(
    interp_frame, selected_distance, selected_distance.get(), *distance
)
distance_dropdown.grid(
    row=1, column=1, columnspan=2, sticky="ew", padx=5, pady=3
)  # Span 2 columns

# Number of Control Points (Interpolation)
control_pts_label = ttk.Label(interp_frame, text="Total Control Pts:")
control_pts_label.grid(row=2, column=0, sticky="w", padx=5, pady=3)
control_pts_valid = root.register(check_control_pts)  # Register with root
control_pts_textbox = ttk.Entry(
    interp_frame, width=10, validate="key", validatecommand=(control_pts_valid, "%P")
)
control_pts_textbox.grid(row=2, column=1, sticky="w", padx=5, pady=3)
control_pts_tooltip = ttk.Label(
    interp_frame, text="(≥ 2, ≤ Data Pts)", font=("Helvetica", 8), foreground="grey"
)
control_pts_tooltip.grid(row=2, column=2, sticky="w", padx=5)


# Bottom Button Frame (Actions)
button_frame = tk.Frame(left_frame)
button_frame.pack(fill=tk.X, pady=10, side=tk.BOTTOM)  # Pack at the bottom

# Use ttk.Button for consistency
ttk.Button(button_frame, text="Load Poses", command=open_file).pack(
    side=tk.LEFT, padx=5, fill=tk.X, expand=True
)
ttk.Button(button_frame, text="Collect Poses", command=set_coords).pack(
    side=tk.LEFT, padx=5, fill=tk.X, expand=True
)
ttk.Button(button_frame, text="Run Motion", command=run_motion).pack(
    side=tk.LEFT, padx=5, fill=tk.X, expand=True
)
ttk.Button(button_frame, text="Release Servos", command=release_servo).pack(
    side=tk.LEFT, padx=5, fill=tk.X, expand=True
)
ttk.Button(button_frame, text="Reset All", command=reset).pack(
    side=tk.LEFT, padx=5, fill=tk.X, expand=True
)


# Right Frame (Plot)
right_frame = tk.Frame(root)
right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)

fig = Figure(figsize=(6, 6), dpi=100)  # Slightly larger figure
ax = fig.add_subplot(111, projection="3d")  # Initialize ax here
ax.set_xlabel("X (m)")
ax.set_ylabel("Y (m)")
ax.set_zlabel("Z (m)")
ax.set_xbound(-0.4, 0.4)
ax.set_ybound(-0.4, 0.4)
ax.set_zlim(0, 0.6)
fig.tight_layout()

canvas = FigureCanvasTkAgg(fig, master=right_frame)  # Initialize canvas
canvas_widget = canvas.get_tk_widget()
canvas_widget.pack(fill=tk.BOTH, expand=True)

toolbar = NavigationToolbar2Tk(canvas, right_frame)
toolbar.update()
# canvas_widget.pack() # canvas already packed above

# Initial Plot Update (after ax, fig, canvas are defined)
update_motion()

# Start GUI Main Loop
root.mainloop()
