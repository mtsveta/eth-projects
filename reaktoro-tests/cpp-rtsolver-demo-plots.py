import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import os

# Auxiliary time related constants
second = 1
minute = 60
hour = 60 * minute
day = 24 * hour
year = 365 * day

# Discretisation parameters
xl = 0.0          # the x-coordinate of the left boundary
xr = 0.5          # the x-coordinate of the right boundary
nsteps = 6000    # the number of steps in the reactive transport simulation
ncells = 50       # the number of cells in the discretization
reltol = 1e-1     # relative tolerance
abstol = 1e-1     # absolute tolerance

D  = 1.0e-9       # the diffusion coefficient (in units of m2/s)
v  = 1.0/day      # the fluid pore velocity (in units of m/s)
dt = minute       # the time step (in units of s)
T = 60.0          # the temperature (in units of K)
P = 100           # the pressure (in units of Pa)

dirichlet = False # the parameter that defines whether Dirichlet BC must be used
smrt_solv = True  # the parameter that defines whether classic or smart
                  # EquilibriumSolver must be used

tag = "-ncells-" + str(ncells) + \
      "-nsteps-" + str(nsteps) + \
      "-reltol-" + "{:.{}e}".format(reltol, 1) + \
      "-abstol-" + "{:.{}e}".format(abstol, 1)

test_tag_smart = tag + "-smart"
test_tag_class = tag + "-reference"

folder_smart   = "../results" + test_tag_smart
folder_class   = "../results" + test_tag_class
folder_results = "results"
folder_general = "results-" + tag

# Output properties 
output_quantities = """
    pH
    speciesMolality(H+)
    speciesMolality(Ca++)
    speciesMolality(Mg++)
    speciesMolality(HCO3-)
    speciesMolality(CO2(aq))
    phaseVolume(Calcite)
    phaseVolume(Dolomite)
""".split()

# Indices of the loaded data to plot
indx_ph        = 0
indx_Hcation   = 1
indx_Cacation  = 2
indx_Mgcation  = 3
indx_HCO3anion = 4
indx_CO2aq     = 5
indx_calcite   = 6
indx_dolomite  = 7

def empty_marker(color):
    return {'facecolor':'white', 'edgecolor':color, 's': 15.0, 'zorder': 2}

def filled_marker(color):
    return {'color':color, 's': 15.0, 'zorder': 2}

def line(color):
    return {'linestyle':'-', 'color':color, 'zorder':1}

def titlestr(t):
    t = t / minute   # Convert from seconds to minutes
    h = int(t) / 60  # The number of hours
    m = int(t) % 60  # The number of remaining minutes
    return 'Time: %2dh %2dm' % (h, m)

def make_results_folders():
    os.system('mkdir -p ' + folder_general + '/figures/ph')
    os.system('mkdir -p ' + folder_general + '/figures/aqueous-species')
    os.system('mkdir -p ' + folder_general + '/figures/calcite-dolomite')
    os.system('mkdir -p ' + folder_general + '/videos')
    os.system('mkdir -p ' + folder_general + '/results')

def get_step(file):
    return int(file.split('.')[0].split('-')[1])

def compare_chemistry(file, status):

    # Get the number of the step
    step = get_step(file)

    # Log the printing of the step
    # print('Plotting figure', step, '...')

    # The current time of the data loaded
    t = step * dt

    # Load the data from the filearray skipping the 1st row
    filearray_smart = np.loadtxt(folder_smart + '/' + file, skiprows=1)
    data_smart = filearray_smart.T
    filearray_class = np.loadtxt(folder_class + '/' + file, skiprows=1)
    data_class = filearray_class.T

    # Number of digits in the total number of the steps
    ndigits = len(str(nsteps))

    # Cells coordinates
    cells = np.linspace(xl, xr, ncells)

    shift = 0.01 * cells[-1]

    # Plot change of pH wrt the space coordinates
    plt.figure()
    plt.title(titlestr(t))
    plt.xlim(left=cells[0]-shift, right=cells[-1]+shift)
    plt.ylim(bottom=2.5, top=10.5)
    plt.title(titlestr(t))
    plt.xlabel('Distance [m]')
    plt.ylabel('pH')
    ph = data_class[indx_ph]
    plt.plot(cells, ph, **line('teal'))
    ph = data_smart[indx_ph]
    # status 0 - training
    # status 1 - prediction
    plt.scatter(cells[status[step]==0], ph[status[step]==0], **empty_marker('teal'))
    plt.scatter(cells[status[step]==0], ph[status[step]==0], **empty_marker('teal'))
    plt.scatter(cells[status[step]==1], ph[status[step]==1], **filled_marker('teal'))
    plt.tight_layout()
    plt.grid(color='lightgray', linestyle=':', linewidth=1)
    plt.savefig(folder_general + '/figures/ph/%s.png' % (str(step).zfill(3)))

    # Plot of mineral's volume the space coordinates
    plt.figure()
    plt.xlim(left=cells[0]-shift, right=cells[-1]+shift)
    plt.ylim(bottom=-0.25, top=3.0+shift)
    plt.title(titlestr(t))
    plt.xlabel('Distance [m]')
    plt.ylabel('Mineral Volume [%$_{\mathsf{vol}}$]')

    data_calcite, data_dolomite = data_class[indx_calcite], data_class[indx_dolomite]
    plt.plot(cells, data_calcite * 100, label='Calcite', **line('indianred'))
    plt.plot(cells, data_dolomite * 100, label='Dolomite', **line('royalblue'))

    data_calcite, data_dolomite  = data_smart[indx_calcite], data_smart[indx_dolomite]
    plt.scatter(cells[status[step]==0], data_calcite[status[step]==0] * 100, **empty_marker('indianred'))
    plt.scatter(cells[status[step]==1], data_calcite[status[step]==1] * 100, **filled_marker('indianred'))
    plt.scatter(cells[status[step]==0], data_dolomite[status[step]==0] * 100, **empty_marker('royalblue'))
    plt.scatter(cells[status[step]==1], data_dolomite[status[step]==1] * 100, **filled_marker('royalblue'))

    plt.legend(loc='center right')
    plt.tight_layout()
    plt.grid(color='lightgray', linestyle=':', linewidth=1)
    plt.savefig(folder_general + '/figures/calcite-dolomite/%s.png' % (str(step).zfill(3)))

    # Plot of aqueous species's concentration the space coordinates
    plt.figure()
    plt.xlim(left=cells[0]-shift, right=cells[-1]+shift)
    plt.ylim(bottom=0.5e-5, top=2)
    plt.yscale('log')
    plt.title(titlestr(t))
    plt.xlabel('Distance [m]')
    plt.ylabel('Concentration [molal]')

    data_cacation  = data_class[indx_Cacation]
    data_mgcation  = data_class[indx_Mgcation]
    data_hco3anion = data_class[indx_HCO3anion]
    data_co2aq     = data_class[indx_CO2aq]
    data_hcation   = data_class[indx_Hcation]

    plt.plot(cells, data_cacation, label='Ca++', **line('steelblue'))
    plt.plot(cells, data_mgcation, label='Mg++', **line('darkorange'))
    plt.plot(cells, data_hco3anion, label='HCO3-',**line('forestgreen'))
    plt.plot(cells, data_co2aq, label='CO2(aq)',**line('red'))
    plt.plot(cells, data_hcation, label='H+', **line('darkviolet'))

    data_cacation  = data_smart[indx_Cacation]
    data_mgcation  = data_smart[indx_Mgcation]
    data_hco3anion = data_smart[indx_HCO3anion]
    data_co2aq     = data_smart[indx_CO2aq]
    data_hcation   = data_smart[indx_Hcation]

    plt.scatter(cells[status[step]==0], data_cacation[status[step]==0], **empty_marker('steelblue'))
    plt.scatter(cells[status[step]==1], data_cacation[status[step]==1], **filled_marker('steelblue'))

    plt.scatter(cells[status[step]==0], data_mgcation[status[step]==0], **empty_marker('darkorange'))
    plt.scatter(cells[status[step]==1], data_mgcation[status[step]==1], **filled_marker('darkorange'))

    plt.scatter(cells[status[step]==0], data_hco3anion[status[step]==0], **empty_marker('forestgreen'))
    plt.scatter(cells[status[step]==1], data_hco3anion[status[step]==1], **filled_marker('forestgreen'))

    plt.scatter(cells[status[step]==0], data_co2aq[status[step]==0], **empty_marker('red'))
    plt.scatter(cells[status[step]==1], data_co2aq[status[step]==1], **filled_marker('red'))

    plt.scatter(cells[status[step]==0], data_hcation[status[step]==0], **empty_marker('darkviolet'))
    plt.scatter(cells[status[step]==1], data_hcation[status[step]==1], **filled_marker('darkviolet'))

    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.grid(color='lightgray', linestyle=':', linewidth=1)

    plt.savefig(folder_general + '/figures/aqueous-species/%s.png' % (str(step).zfill(3)))

    plt.close('all')

def plot_cpu_times(files):

    data = [[], [], [], []]
    indx_rt_smart = 0
    indx_eq_smart = 1
    indx_rt_class = 2
    indx_eq_class = 3
    indx = 1e16

    # Load the data from the filearray skipping the 1st row
    for file in files:
        file_name = file[0]
        is_smart  = file[1]

        folder = folder_smart if is_smart else folder_class
        data_ = np.loadtxt(folder + '/' + file_name)
        if  "RT" in file_name and is_smart == 1:
            indx = indx_rt_smart
        elif "EQ" in file_name and is_smart == 1:
            indx = indx_eq_smart
        elif "RT" in file_name and is_smart == 0:
            indx = indx_rt_class
        elif "EQ" in file_name and is_smart == 0:
            indx = indx_eq_class
        data[indx] = data_.T
    time = np.linspace(0, dt * nsteps / minute, nsteps + 1)

    # Plot change of pH wrt the space coordinates
    plt.figure()
    plt.xlim(left=0, right=time[-1])
    #$plt.ylim(bottom=2.5, top=10.5)
    #plt.title(titlestr(t))
    plt.xlabel('Time (minute)')
    plt.ylabel('CPU time (seconds)')
    plt.plot(time, data[indx_eq_class], label="Equlibrium (Conventional)",
             color='steelblue')
    plt.plot(time, data[indx_eq_smart], label="Equlibrium (Smart)",
             color='darkorange')
    plt.plot(time, data[indx_rt_class], label="Reactive transport",
             color='forestgreen')
    plt.grid(color='lightgray', linestyle=':', linewidth=1)
    plt.legend(loc='upper right')
    plt.savefig(folder_general + '/figures/cpu-time.png')

def count_trainings(status):
    counter = [st for st in status if st == 0]
    return (len(counter), len(status))

if __name__ == '__main__':

    profiling = []
    status    = []

    make_results_folders()

    # Collect files with results corresponding to smart or reference (classical) solver
    files_smart = [file for file in sorted(os.listdir(folder_smart)) if ("profiling" not in file) and ("tracker" not in file)]
    files_class = [file for file in sorted(os.listdir(folder_class)) if ("profiling" not in file) and ("tracker" not in file)]

    # Load the status data, where 0 stands for conventional learning and 1 for smart prediction
    status_file = [file for file in sorted(os.listdir(folder_smart)) if "status" in file]
    if status_file != []: status = np.loadtxt(folder_smart + '/' + status_file[0])

    # Collect files with profiling data
    profiling   = profiling + [(file, 1) for file in sorted(os.listdir(folder_smart)) if "profiling" in file]
    profiling   = profiling + [(file, 0) for file in sorted(os.listdir(folder_class)) if "profiling" in file]

    # Plot profiling results
    plot_cpu_times(profiling)

    # Count the percentage of the trainings needed
    training_counter = count_trainings(np.array(status).flatten())
    print("%2.2f percent is training (%d learnings out of %d simulations)" % (100 * training_counter[0] / training_counter[1], training_counter[0], training_counter[1]))

    # Plot comparison of the chemistry of the conventional and smart solvers
    Parallel(n_jobs=16)(delayed(compare_chemistry)(file, status) for file in files_smart)

    # Save selected plots into results folder
    selected_steps = [10, 60, 1200, 2400, 4800]
    for step in selected_steps:
        copy_cmd = ('cp -i ' + folder_general + '/figures/{0}/%s.png ' + folder_general + '/results/{0}-%s-%dmin.png') % (str(step).zfill(3), str(step).zfill(3), step*dt/minute)
        os.system(copy_cmd.format('calcite-dolomite'))
        os.system(copy_cmd.format('ph'))
        os.system(copy_cmd.format('aqueous-species'))

    # Make  videos
    # --------------------------------------------------------------------------
    # Define the command for making videos
    ffmpeg_cmd = 'ffmpeg -y -r 30 -i ' + folder_general + '/figures/{0}/%03d.png ' \
                 '-codec:v mpeg4 -flags:v +qscale -global_quality:v 0 ' \
                 + folder_general + '/videos/{0}.mp4'
    os.system(ffmpeg_cmd.format('calcite-dolomite'))
    os.system(ffmpeg_cmd.format('aqueous-species'))
    os.system(ffmpeg_cmd.format('ph'))

    """
    for file in files_smart:
        compare_chemistry(file, status);
    """
'''
    # Plot all result files
    for is_smart in range(1, 2):
        # Defined the tag based on the eq. solder
        tag = "1" if is_smart == True else "0"

        # Create folders for the plots and videos with
        make_results_folders(tag)

        # Defined the folder based on the eq. solver and load the files
        folder = folder_smart if is_smart else folder_class
        files = sorted(os.listdir(folder))

        # Collect files for profiling
        profiling   = profiling + [(file, is_smart) for file in files if
                      "profiling" in file]
        status_file = [file for file in files if is_smart and "status" in file]
        if status_file != []:
            status = loadtxt(folder + '/' + status_file[0])

        for file in files[0:5]:
            if ("profiling" not in file) and ("tracker" not in file):
                plot_chemistry(file, folder, is_smart, status);
        
        Parallel(n_jobs=16)(delayed(plot_chemistry)(file, folder, is_smart, status)
                            for file in files if "profiling" not in file)

        # Define the command for making videos
        ffmpeg_cmd = 'ffmpeg -y -r 30 -i figures/{0}-' + tag + '/%03d.png ' \
                     '-codec:v mpeg4 -flags:v +qscale -global_quality:v 0 ' \
                     'videos/{0}-' + tag + '.mp4'
        os.system(ffmpeg_cmd.format('calcite-dolomite'))
        os.system(ffmpeg_cmd.format('aqueous-species'))
        os.system(ffmpeg_cmd.format('ph'))
        
'''

