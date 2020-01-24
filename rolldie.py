"""
Demonstrate matplotlib animation of random values.

Use a uniformly distributed random number generator
in range(1, 7) to simulate die rolls. Dynamically
graph frequencies of the die faces.


R. Mak
Department of Applied Data Science
San Jose State University
"""

from matplotlib import animation
import matplotlib.pyplot as plt
import random
import seaborn as sns
import sys


def show_bar_chart(title, x_label, y_label, x_values, y_values, bar_toppers):
    """
    Display a bar chart.
    @param title the chart title.
    @param x_label the label for the x axis
    @param y_label the label for the y axis
    @param x_values the x values to plot
    @param y_values the y values to plot
    @param bar_toppers the text above each bar
    """

    plt.cla()  # clear old contents

    axes = sns.barplot(x_values, y_values, color='green')
    axes.set_title(title)
    axes.set(xlabel=x_label, ylabel=y_label)

    # Scale the y-axis by 10% to make room for text above the bars.
    axes.set_ylim(top=1.10 * max(y_values))

    # Display the topper text above each patch (bar).
    for bar, topper in zip(axes.patches, bar_toppers):
        text_x = bar.get_x() + bar.get_width() / 2
        text_y = bar.get_height()
        axes.text(text_x, text_y, topper,
                  fontsize=11, ha='center', va='bottom')


def update_frame(frame_number, roll_count, faces, frequencies):
    """
    Update the bar plot contents for each animation frame.
    @param frame_number the frame number.
    @param roll_count the count of die rolls per frame.
    @param faces the die faces.
    @param frequencies the list of die face frequencies.
    """

    # Roll the die 'roll_count' times and update the face frequencies.
    for _ in range(roll_count):
        frequencies[random.randrange(1, 7) - 1] += 1

        # Set the percentages for the bar tops.
    freq_sum = sum(frequencies)
    topper = [f'{freq:,}\n{freq / freq_sum:.3%}' for freq in frequencies]

    # Display the bar chart for this frame.
    show_bar_chart(f'Face Frequencies for {freq_sum:,} Die Rolls ' +
                   f'(Frame {frame_number + 2})',
                   'Face Value', 'Frequency',
                   faces, frequencies, topper)


# Read command-line arguments for the number of frames
# and the number of rolls per frame.
number_of_frames = int(sys.argv[1])
rolls_per_frame = int(sys.argv[2])

# Create the figure for the animation.
sns.set_style('whitegrid')  # white background with gray grid lines
figure = plt.figure('Rolling a Six-Sided Die')

values = list(range(1, 7))  # die faces for display on the x-axis
frequencies = [0] * 6  # six-element list of die frequencies

# Configure and start the animation that calls function update_frame.
die_animation = animation.FuncAnimation(
    figure, update_frame, repeat=False,
    frames=number_of_frames - 1, interval=25,
    fargs=(rolls_per_frame, values, frequencies))

plt.show()

# Adapted from:
# **************************************************************************
# * (C) Copyright 1992-2018 by Deitel & Associates, Inc. and               *
# * Pearson Education, Inc. All Rights Reserved.                           *
# *                                                                        *
# * DISCLAIMER: The authors and publisher of this book have used their     *
# * best efforts in preparing the book. These efforts include the          *
# * development, research, and testing of the theories and programs        *
# * to determine their effectiveness. The authors and publisher make       *
# * no warranty of any kind, expressed or implied, with regard to these    *
# * programs or to the documentation contained in these books. The authors *
# * and publisher shall not be liable in any event for incidental or       *
# * consequential damages in connection with, or arising out of, the       *
# * furnishing, performance, or use of these programs.                     *
# **************************************************************************