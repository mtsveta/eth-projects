from numpy import *
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
xr = 100.0          # the x-coordinate of the right boundary
nsteps = 100      # the number of steps in the reactive transport simulation
ncells = 100      # the number of cells in the discretization

D  = 1.0e-9       # the diffusion coefficient (in units of m2/s)
v  = 1.0/day      # the fluid pore velocity (in units of m/s)
dt = 10*minute    # the time step (in units of s)
T = 60.0 + 273.15 # the temperature (in units of K)
P = 100 * 1e5     # the pressure (in units of Pa)

dirichlet = False # the parameter that defines whether Dirichlet BC must be used
smrt_solv = True  # the parameter that defines whether classic or smart
                  # EquilibriumSolver must be used

test_tag_smart = "-ncells-" + str(ncells) + \
                 "-nsteps-" + str(nsteps) + \
                 "-ismart-" + "1"

test_tag_class = "-ncells-" + str(ncells) + \
                 "-nsteps-" + str(nsteps) + \
                 "-ismart-" + "0"

folder_smart = "../results" + test_tag_smart
folder_class = "../results" + test_tag_class

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

colors = ['aqua', 'darkblue',
          'coral', 'crimson',
          'green', 'darkgreen',
          'orange', 'darkorange',
          'pink', 'maroon',
          'gold', 'brown']

lines   = ['', 'solid']
markers = ['o', '']
markerfacecolor='orange'

#facecolors='none'

def titlestr(t):
    t = t / minute   # Convert from seconds to minutes
    h = int(t) / 60  # The number of hours
    m = int(t) % 60  # The number of remaining minutes
    return 'Time: %2d h %2d m' % (h, m)

def make_results_folders(tag):
    os.system('mkdir -p figures/ph-' + tag)
    os.system('mkdir -p figures/aqueous-species-' + tag)
    os.system('mkdir -p figures/calcite-dolomite-' + tag)
    os.system('mkdir -p videos')

def plot_chemistry(file, folder, is_smart):

    # Define tag based on the equilibrium solver
    tag = "1" if is_smart == True else "0"

    # Get the number of the step
    step = int(file.split('.')[0].split('-')[1])

    # Log the printing of the step
    print('Plotting figure', step, '...')

    # The current time of the data loaded
    t = step * dt

    # Load the data from the filearray skipping the 1st row
    filearray = loadtxt(folder + '/' + file, skiprows=1)
    data = filearray.T

    # Number of digits in the total number of the steps
    ndigits = len(str(nsteps))

    # Cells coordinates
    cells = linspace(xl, xr, ncells)

    if tag == "1":
        marker = 'o'
        line   = ''
    else:
        marker = ''
        line   = 'solid'

    # Plot change of pH wrt the space coordinates 
    plt.figure()
    #plt.xlim(left=-0.02, right=0.52)
    #plt.ylim(bottom=2.5, top=10.5)
    plt.title(titlestr(t))
    plt.xlabel('Distance [m]')
    plt.ylabel('pH')
    plt.plot(cells, data[indx_ph],
             linestyle=line,
             color=colors[0],
             marker=marker,
             markerfacecolor=colors[1] if is_smart else 'white')
    #plt.show()
    #plt.tight_layout()
    plt.savefig('figures/ph-' + tag + '/{}.png'.format(str(step).zfill(
        ndigits)))

    # Plot of mineral's volume the space coordinates 
    plt.figure()
    #plt.xlim(left=-0.02, right=0.52)
    #plt.ylim(bottom=-0.1, top=2.1)
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    plt.title(titlestr(t))
    plt.xlabel('Distance [m]')
    plt.ylabel('Mineral Volume [%$_{\mathsf{vol}}$]')
    plt.plot(cells, data[indx_calcite] * 100, label='Calcite',
             linestyle=line,
             color=colors[1],
             marker=marker,
             markerfacecolor=colors[1] if is_smart else 'white')
    plt.plot(cells, data[indx_dolomite] * 100, label='Dolomite',
             linestyle=line,
             color=colors[3],
             marker=marker,
             markerfacecolor=colors[3] if is_smart else 'white')
    plt.legend(loc='center right')
    #plt.show()
    #plt.tight_layout()
    plt.savefig('figures/calcite-dolomite-' + tag + '/{}.png'.format(str(step).zfill(
        ndigits)))

    plt.figure()
    plt.yscale('log')
    #plt.xlim(left=-0.02, right=0.52)
    #plt.ylim(bottom=0.5e-5, top=2)
    plt.title(titlestr(t))
    plt.xlabel('Distance [m]')
    plt.ylabel('Concentration [molal]')
    plt.plot(cells, data[indx_Cacation], label='Ca++',
             linestyle=line,
             color=colors[2],
             marker=marker,
             markerfacecolor=colors[2] if is_smart else 'white')
    plt.plot(cells, data[indx_Mgcation], label='Mg++',
             linestyle=line,
             color=colors[4],
             marker=marker,
             markerfacecolor=colors[4] if is_smart else 'white')
    plt.plot(cells, data[indx_HCO3anion], label='HCO3-',
             linestyle=line,
             color=colors[5],
             marker=marker,
             markerfacecolor=colors[5] if is_smart else 'white')
    plt.plot(cells, data[indx_CO2aq], label='CO2(aq)',
             linestyle=line,
             color=colors[6],
             marker=marker,
             markerfacecolor=colors[6] if is_smart else 'white')
    plt.plot(cells, data[indx_Hcation], label='H+',
             linestyle=line,
             color=colors[7],
             marker=marker,
             markerfacecolor=colors[7] if is_smart else 'white')
    plt.legend(loc='lower right')
    #plt.show()
    #plt.tight_layout()
    plt.savefig('figures/aqueous-species-' + tag + '/{}.png'.format(str(step).zfill(ndigits)))

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
        data_ = loadtxt(folder + '/' + file_name)
        if  "RT" in file_name and is_smart == 1:
            indx = indx_rt_smart
        elif "EQ" in file_name and is_smart == 1:
            indx = indx_eq_smart
        elif "RT" in file_name and is_smart == 0:
            indx = indx_rt_class
        elif "EQ" in file_name and is_smart == 0:
            indx = indx_eq_class
        data[indx] = data_.T
    time = linspace(0, dt * nsteps / minute, nsteps + 1)

    # Plot change of pH wrt the space coordinates
    plt.figure()
    #plt.xlim(left=-0.02, right=0.52)
    #$plt.ylim(bottom=2.5, top=10.5)
    #plt.title(titlestr(t))
    plt.xlabel('Time (minute)')
    plt.ylabel('CPU time (seconds)')
    plt.plot(time, data[indx_rt_class], label="Reactive transport",
             color=colors[1])
    plt.plot(time, data[indx_eq_smart], label="Equlibrium (Smart)",
             color=colors[3])
    plt.plot(time, data[indx_eq_class], label="Equlibrium (Conventional)",
             color=colors[5])
    plt.legend(loc='upper left')
    plt.show()
    plt.savefig('figures/cpu-time.png')

if __name__ == '__main__':

    profiling = []

    # Plot all result files
    for is_smart in range(2):
        # Defined the tag based on the eq. solder
        tag = "1" if is_smart == True else "0"
        # Create folders for the plots and videos with
        make_results_folders(tag)
        # Defined the folder based on the eq. solver and load the files
        folder = folder_smart if is_smart else folder_class
        files = sorted(os.listdir(folder))

        # Collect files for profiling
        profiling = profiling + [(file, is_smart) for file in files if
                     "profiling" in file]

        '''
        Parallel(n_jobs=16)(delayed(plot_chemistry)(file, folder, is_smart)
                            for file in files if "profiling" not in file)

        # Define the command for making videos
        ffmpeg_cmd = 'ffmpeg -y -r 30 -i figures/{0}-' + tag + '/%03d.png ' \
                     '-codec:v mpeg4 -flags:v +qscale -global_quality:v 0 ' \
                     'videos/{0}-' + tag + '.mp4'
        os.system(ffmpeg_cmd.format('calcite-dolomite'))
        os.system(ffmpeg_cmd.format('aqueous-species'))
        os.system(ffmpeg_cmd.format('ph'))
        '''
    # Plot profiling results
    plot_cpu_times(profiling)

