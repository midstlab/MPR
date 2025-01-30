import numpy as np

def calculate_final_coords_and_generate_bild(initial_coords, vectors, output_file, scale_factor=20, arrow_radius=0.2):
    """
    Calculate final coordinates and generate a .bild file for ChimeraX arrows.
    
    Parameters:
    - initial_coords (list of lists): List of initial coordinates for each arrow.
    - vectors (list of lists): List of direction vectors for each arrow.
    - output_file (str): Path to the output .bild file.
    - scale_factor (int): Value to divide the vectors by (default: 10).
    - arrow_radius (float): Radius of the arrows in the .bild file (default: 0.2).
    """
    # Convert input lists to numpy arrays
    initial_coords = np.array(initial_coords)
    vectors = np.array(vectors)
    
    # Scale the vectors
    scaled_vectors = vectors / scale_factor
    
    # Calculate final coordinates
    final_coords = initial_coords + scaled_vectors
    
    # Generate .bild content
    bild_content = []
    for i, (start, end, vector, scaled_vector) in enumerate(zip(initial_coords, final_coords, vectors, scaled_vectors)):
        bild_content.append(f"# Arrow {i + 1}\n")
        bild_content.append(f"# Initial (x1, y1, z1): {start[0]:.3f} {start[1]:.3f} {start[2]:.3f}\n")
        bild_content.append(f"# Vector: {vector[0]:.3f} {vector[1]:.3f} {vector[2]:.3f}\n")
        bild_content.append(f"# Scaled Vector: {scaled_vector[0]:.3f} {scaled_vector[1]:.3f} {scaled_vector[2]:.3f}\n")
        bild_content.append(f"# Final (x2, y2, z2): {end[0]:.3f} {end[1]:.3f} {end[2]:.3f}\n")
        bild_content.append(".color red\n")
        bild_content.append(
            f".arrow {start[0]:.3f} {start[1]:.3f} {start[2]:.3f} "
            f"{end[0]:.3f} {end[1]:.3f} {end[2]:.3f} {arrow_radius}\n"
        )
    
    # Write to the output file
    with open(output_file, 'w') as file:
        file.writelines(bild_content)

    print(f"BILD file saved to {output_file}")


# Example usage
initial_coords = [
    [22.000,  -1.471, -24.891],
    [12.562,  -6.598, -28.828],
    [-2.062,  -4.539,  -3.693],
    [-29.172,  -3.037,   4.184],
    [-26.000,   0.957,  22.203]
]
vectors = [
    [-48.143595998234645, 2.4495041730190548, 140.29353735663022],
    [-26.746729630902795, 21.299502376152034, 140.62052649378913],
    [113.7270087440215, 11.664611766753561, -117.50185032368921],
    [70.7280256637552, 26.877321540505275, 47.65991359105376],
    [38.45725284337522, -4.128591740516413, -101.46280971191713]
]
output_file = "output_arrows_with_vectors5.bild"

# Run the function
calculate_final_coords_and_generate_bild(initial_coords, vectors, output_file)

