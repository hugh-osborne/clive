# Import ARRM (Afferent Responsive Reduced Model?) - An OpenSim upper arm model
import arrm.arrm as arrm
# Import MIIND for neural control (miindsim for CPU, miindsimv for GPGPU)
# Note that MIIND can be replaced with any other (or your own) neural simulator.
import miind.miindsimv as miind

import numpy as np
import csv
from math import comb
import itertools
import matplotlib.pyplot as plt

def basic_simulation_example():
    time_step = 0.0002 # Recommended time step is 0.0002. Greater than this and OpenSim can blowup
    sim_time = 0.1 #seconds
    render_the_model = True
    
    # Initialise the ARRM class
    m = arrm.ARRMSim(time_step, sim_time, render_the_model)
    
    # Initialise the MIIND class
    # lif.xml holds a MIIND simulation with a Leaky integrate and fire population for each muscle in the
    # opensim model. 
    # lif.xml also defines the timestep of the MIIND simulation which, for simplicity, should match the 
    # time_step of ARRM (0.0002). Otherwise, there's going to be trouble. 
    miindsim = miind.init("lif.xml", TIME_END=str(sim_time))
    
    # Load the simulation and geometry files required by OpenSim
    arrm.loadResources(m)

    # Here we can enabled or disable each muscle.
    # Disabled muscles (m.*active = False) will be held fixed at the default length i.e. not floppy!
    # In this example, FCR and ECR are disabled so the wrist will remain locked.
    # The lif.xml simulation file has one population per *active* muscle. I.e. if you want to enabled fcr or ecr,
    # you need to add those populations to the xml file.
    m.biceps_long_active = True
    m.brachioradialis_active = True
    m.triceps_long_active = True
    m.triceps_med_active = True
    m.delt_ant_active = True
    m.delt_med_active = True
    m.delt_post_active = True
    m.infraspinatus_active = True
    m.pec_maj_active = True
    m.lat_active = True
    m.fcr_active = False
    m.ecr_active = False

    # The following is the list of enumerations which can be used to identify each muscle in ARRM
    # These are needed for functions later when reading or writing muscle activity to ARRM
    # m.biceps_long
    # m.brachioradialis
    # m.triceps_long
    # m.triceps_med
    # m.delt_ant
    # m.delt_med
    # m.delt_post
    # m.infraspinatus
    # m.pec_maj
    # m.lat
    # m.fcr
    # m.ecr
    
    # A string which connects the hand to a fixed point below and behind the elbow's initial position
    # can be enabled (Used for reproducing Raf's experiments)
    m.string_active = False
    m.string_length = 1.55
    
    # A rod which connects the hand to a fixed point below and behind the elbow's inital position
    # can be enabled. Unlike the string, the rod cannot be compressed.
    m.rod_active = False
    m.rod_length = 0.50
    
    # The hand can also be held in its initial position allowing force vectors to be calculated.
    # The arm will remain relatively stable even with individual muscles activated.
    m.point_constraint_active = False
    
    # The individual joints of ARRM can be enabled and disabled and set at initial angles.
    # Disabled joints will be held at the defined start angle but are still subject enabled
    # joints higher up the arm.
    # Angles are in radians
    m.locked_shoulder_add_abd = False
    m.start_angle_shoulder_add_abd = 0.0

    m.locked_shoulder_ext_flex = False
    m.start_angle_shoulder_ext_flex = 0.0

    m.locked_shoulder_twist = False
    m.start_angle_shoulder_twist = -0.698132

    m.locked_elbow = False
    m.start_angle_elbow = 0.698132*2
    
    # Begin the ARRM simulation
    m.begin()
    # Begin the MIIND simulation
    miind.startSimulation()

    # It is useful to be able to track the hand position
    init_hand_x = m.getHandPositionX()
    init_hand_y = m.getHandPositionY()
    init_hand_z = m.getHandPositionZ()

    # It is useful to be able to track the shoulder angles
    init_shoulder_abd_value = m.getShoulderAbdAngle()
    init_shoulder_twist_value = m.getShoulderTwistAngle()
    
    # Other values available. These can all be called during the simulation and recorded.
    # m.getHandPositionX()
	# m.getHandPositionY()
	# m.getHandPositionZ()

	# m.getElbowPositionX()
	# m.getElbowPositionY()
	# m.getElbowPositionZ()

	# m.getElbowAngle()
	# m.getShoulderAbdAngle()
	# m.getShoulderTwistAngle()
	# m.getShoulderExtAngle()

	# m.getHandAccelX()
	# m.getHandAccelY()
	# m.getHandAccelZ()
    
    num_enabled_muscles = 10

    # Example cortical input: Each population in MIIND will recieve a 30Hz signal
    # Remember that this could be any number of synapses (e.g 10 corticospinal synapses @ 3Hz)
    cortical_input = [30.0 for a in range(num_enabled_muscles)]
    # The order of the values in the cortical_input array must match the order of the 
    # IncomingConnections defined in 
    # the lif.xml file. To avoid confusion, this should probably follow the order of muscles
    # defined above.
    # 30Hz isn't enough to get the MIIND populations (and therefore the muscles) to fire.
    # Let's add an additional 60Hz to the biceps population (the first population).
    # We expect to see the forearm lift.
    cortical_input[0] += 60.0

    # For this example, we will track the Ia and Ib afferent signals of each active muscle
    # We'll also use the ias to set up a rudimentary stretch reflex in the triceps muscle
    # to offset the biceps activity.
    ias = [0.0 for a in range(num_enabled_muscles)]
    ibs = [0.0 for a in range(num_enabled_muscles)]
    
    # Let's also record the position and acceleration of the hand
    hand_pos = []
    hand_accel = []

    # Run the correct number of iterations for the required simulation time
    for i in range(int(sim_time / time_step)): 
        # Record the hand position
        hand_pos = hand_pos + [[m.getHandPositionX(),m.getHandPositionY(),m.getHandPositionZ()]]

        # Record the hand acceleration
        hand_accel = hand_accel + [[m.getHandAccelX(), m.getHandAccelY(), m.getHandAccelZ()]]
        
        # Get the last triceps Ia signal. This has no units (it's probably not Hz) so
        # it needs adjusting before being added to the triceps population input (muscle id = 2)
        triceps_ia = ias[2] * 0.05 # ias[1] stores the Ia afferent activity from the previous iteration. Choose 0.05 as a scaling factor so that the signal isn't too strong.
        
        # Classic stretch reflex is an excitatory effect on the homonymous muscle from Ia
        # so let's add triceps_ia to the triceps cortical input
        cortical_input[2] += triceps_ia
        
        # While we are adding and possibly subtracting activities to each population, make sure that,
        # in the end, the input to each muscle is above 0.
        cortical_input = [max(a,0) for a in cortical_input]
        
        # This performs the MIIND iteration. Activities is a list of population output firing rates
        # The order matches that of the OutgoingConnections defined in lif.xml. Again, to avoid confusion,
        # this should match the order of muscle defined above.
        activities = miind.evolveSingleStep(cortical_input)
        
        # Optionally, we can limit the output firing rates so they don't make the muscles
        # go crazy.
        activities = [min(a,50) for a in activities]
        
        # Further changes can be made to the population firing rates here if required.
        # Although the firing rates are measured in Hz, the muscles in ARRM respond
        # to an "activity" level which is not measured in Hz. As with the Ia signal above,
        # some scale factor may be required to get an expected activation.
        # Future work for this system would be to precisely calculate what those scale-factors 
        # should be.
        
        # The firing rates of the neural populations must now be passed to the muscles in ARRM.
        # First parameter is the identifier of the muscle (defined above)
        # Second, third, and fourth parameters are the Alpha, Beta, and Gamma signals respectively
        # to be passed to the given muscle.
        m.setMuscleActivity(m.biceps_long, activities[0], 0.0, 5000.0)
        m.setMuscleActivity(m.brachioradialis, activities[1], 0.0, 5000.0)
        m.setMuscleActivity(m.triceps_long, activities[2], 0.0, 5000.0)
        m.setMuscleActivity(m.triceps_med, activities[3], 0.0, 5000.0)
        m.setMuscleActivity(m.delt_ant, activities[4], 0.0, 5000.0)
        m.setMuscleActivity(m.delt_med, activities[5], 0.0, 5000.0)
        m.setMuscleActivity(m.delt_post, activities[6], 0.0, 5000.0)
        m.setMuscleActivity(m.infraspinatus, activities[7], 0.0, 5000.0)
        m.setMuscleActivity(m.pec_maj, activities[8], 0.0, 5000.0)
        m.setMuscleActivity(m.lat, activities[9], 0.0, 5000.0)
        
        # Perform the ARRM iteration.
        m.update()
        
        # Record the Ia and Ib signals for each muscle for use in the next iteration
        ias[0] = m.getMuscleIa(m.biceps_long)
        ias[1] = m.getMuscleIa(m.brachioradialis)
        ias[2] = m.getMuscleIa(m.triceps_long)
        ias[3] = m.getMuscleIa(m.triceps_med)
        ias[4] = m.getMuscleIa(m.delt_ant)
        ias[5] = m.getMuscleIa(m.delt_med)
        ias[6] = m.getMuscleIa(m.delt_post)
        ias[7] = m.getMuscleIa(m.infraspinatus)
        ias[8] = m.getMuscleIa(m.pec_maj)
        ias[9] = m.getMuscleIa(m.lat)
        
        ibs[0] = m.getMuscleIb(m.biceps_long)
        ibs[1] = m.getMuscleIb(m.brachioradialis)
        ibs[2] = m.getMuscleIb(m.triceps_long)
        ibs[3] = m.getMuscleIb(m.triceps_med)
        ibs[4] = m.getMuscleIb(m.delt_ant)
        ibs[5] = m.getMuscleIb(m.delt_med)
        ibs[6] = m.getMuscleIb(m.delt_post)
        ibs[7] = m.getMuscleIb(m.infraspinatus)
        ibs[8] = m.getMuscleIb(m.pec_maj)
        ibs[9] = m.getMuscleIb(m.lat)

    # Finish the MIIND and ARRM simulations.
    miind.endSimulation()
    m.end()
    
    # Let's plot the Y hand position and acceleration over the duration of the simulation
    # Initially, the hand drops because of both gravity and that triceps is already stretched 
    # so begins pulling it down. Eventually, the biceps activation overtakes it and the acceleration reverses.
    fig, ax1 = plt.subplots()
    colorE = '#0000FF'
    ax1.set_xlabel('Time (s)', size=14)
    ax1.set_ylabel('Y Position/Acceleration', size=14)
    line1 = ax1.plot([a*time_step for a in range(int(sim_time / time_step))], [(h[1]-hand_pos[0][1])*1000 for h in hand_pos])
    line1 = ax1.plot([a*time_step for a in range(int(sim_time / time_step))], [h[1] for h in hand_accel])
    fig.tight_layout()
    plt.show()
    
    # The simulation has also generated a number of results files
    
    

def doit(pattern, simtime=0.1, return_accel = False, rod = True, vis = False, control=True, use_ias=True):
    time_step = 0.0002
    sim_time = simtime
    m = arrm.ARRMSim(time_step, sim_time, True)

    miindsim = miind.init("lif.xml", TIME_END=str(sim_time))

    miind_time_step = miind.getTimeStep();
    num_miind_steps_per_arrm_step = int(time_step / miind_time_step)
    num_miind_steps_per_arrm_step = 1

    
   

    m.begin()
    miind.startSimulation()

    init_hand_x = m.getHandPositionX()
    init_hand_y = m.getHandPositionY()
    init_hand_z = m.getHandPositionZ()

    init_shoulder_abd_value = m.getShoulderAbdAngle()
    init_shoulder_twist_value = m.getShoulderTwistAngle()
    
    num_muscles = 10

    cortical_input = [30.0 + pattern[a]*40 for a in range(num_muscles)]

    ias = [0.0 for a in range(num_muscles)]
    ibs = [0.0 for a in range(num_muscles)]
    
    accels = []

    for i in range(int(sim_time / time_step)):
        if (i % num_miind_steps_per_arrm_step == 0):
            
            balance = [0.0 for k in range(num_muscles)]
            shoulder_abd_value = m.getShoulderAbdAngle()
            shoulder_twist_value = m.getShoulderTwistAngle()

            balance[5] += ((shoulder_abd_value - init_shoulder_abd_value) * 50.0)
            balance[6] += ((shoulder_abd_value - init_shoulder_abd_value) * 100.0)
            
            balance[3] += ((init_shoulder_abd_value - shoulder_abd_value) * 100.0)
            
            balance[4] += ((shoulder_twist_value - init_shoulder_twist_value) * 50.0)
            balance[0] += (-(shoulder_twist_value - init_shoulder_twist_value) * 50.0)
            balance[1] += ((shoulder_twist_value - init_shoulder_twist_value) * 50.0)

            balance[5] += (((init_shoulder_twist_value - shoulder_twist_value)) * 20.0)
            balance[0] += ((shoulder_twist_value - init_shoulder_twist_value) * 50.0)
            balance[1] += (-(shoulder_twist_value - init_shoulder_twist_value) * 50.0)

            hand_x = m.getHandPositionX()
            hand_y = m.getHandPositionY()
            hand_z = m.getHandPositionZ()

            balance[0] += ((init_hand_y - hand_y) * 50.0)
            balance[1] += ((hand_y - init_hand_y) * 50.0)
            
            if return_accel:
                accels = accels + [[m.getHandAccelX(), m.getHandAccelY(), m.getHandAccelZ()]]
            
            balance_mult = 0.0
            if control:
                balance_mult = 5.0
            ias_mult = 0.0
            ibs_mult = 0.0
            if use_ias:
                ias_mult = 0.2
                ibs_mult = 0.0
                
            total_input = [cortical_input[k] for k in range(num_muscles)]
            activities = miind.evolveSingleStep([max(a,0) for a in total_input])
            activities = [min(a,100) for a in activities]
            
            
        m.setMuscleActivity(m.biceps_long, activities[0], 0.0, 5000.0)
        m.setMuscleActivity(m.brachioradialis, activities[1], 0.0, 5000.0)
        m.setMuscleActivity(m.triceps_long, activities[2], 0.0, 5000.0)
        m.setMuscleActivity(m.triceps_med, activities[3], 0.0, 5000.0)
        m.setMuscleActivity(m.delt_ant, activities[4], 0.0, 5000.0)
        m.setMuscleActivity(m.delt_med, activities[5], 0.0, 5000.0)
        m.setMuscleActivity(m.delt_post, activities[6], 0.0, 5000.0)
        m.setMuscleActivity(m.infraspinatus, activities[7], 0.0, 5000.0)
        m.setMuscleActivity(m.pec_maj, activities[8], 0.0, 5000.0)
        m.setMuscleActivity(m.lat, activities[9], 0.0, 5000.0)
        
        m.update()
        ias[0] = m.getMuscleIa(m.biceps_long)
        ias[1] = m.getMuscleIa(m.brachioradialis)
        ias[2] = m.getMuscleIa(m.triceps_long)
        ias[3] = m.getMuscleIa(m.triceps_med)
        ias[4] = m.getMuscleIa(m.delt_ant)
        ias[5] = m.getMuscleIa(m.delt_med)
        ias[6] = m.getMuscleIa(m.delt_post)
        ias[7] = m.getMuscleIa(m.infraspinatus)
        ias[8] = m.getMuscleIa(m.pec_maj)
        ias[9] = m.getMuscleIa(m.lat)
        
        ibs[0] = m.getMuscleIb(m.biceps_long)
        ibs[1] = m.getMuscleIb(m.brachioradialis)
        ibs[2] = m.getMuscleIb(m.triceps_long)
        ibs[3] = m.getMuscleIb(m.triceps_med)
        ibs[4] = m.getMuscleIb(m.delt_ant)
        ibs[5] = m.getMuscleIb(m.delt_med)
        ibs[6] = m.getMuscleIb(m.delt_post)
        ibs[7] = m.getMuscleIb(m.infraspinatus)
        ibs[8] = m.getMuscleIb(m.pec_maj)
        ibs[9] = m.getMuscleIb(m.lat)

    miind.endSimulation()
    m.end()
    
    if not return_accel:
        with open("jointreaction_Un-named analysis._ReactionLoads") as jrfile:

            reader = csv.reader(jrfile, delimiter='\t', quotechar='|')
            i=0
            
            hand_mass = 0.00001
            gravity = 9.80665
            
            next_time = 0.1
            x = 0
            y = 0
            z = 0
            num_rows_read = 0
            for row in reader:
                if i > 15 and float(row[0]) > next_time:
                    next_time += 0.01
                    start_index = 1
                    x += -float(row[start_index])/hand_mass
                    y += -float(row[start_index+1])/hand_mass
                    z += -float(row[start_index+2])/hand_mass
                    num_rows_read += 1
                    
                i += 1
                
            return x*hand_mass/num_rows_read,y*hand_mass/num_rows_read,z*hand_mass/num_rows_read
    else:
        accels = np.array(accels)
        return np.mean(accels[int(accels.shape[0]/2):,0]),np.mean(accels[int(accels.shape[0]/2):,1]),np.mean(accels[int(accels.shape[0]/2):,2])

def sweep(filename="force_results.csv", start_trial=0, problem_trials=[]):
    
    num_muscles = 10
    
    form = 'w'
    if start_trial > 0:
        form = 'a'
        
    with open(filename, form) as results_file:
        
        results_writer = csv.writer(results_file, delimiter=',', lineterminator='\n')
        
        tests = []
        trial_count = 0
        for d1 in range(num_muscles):
            for d2 in [a for a in itertools.combinations(range(num_muscles),d1)]:
                trial_count += 1
                
                pattern = [1 if a in d2 else 0 for a in range(num_muscles)]
                
                if (trial_count < start_trial):
                    continue
                
                if (trial_count in problem_trials):
                    results_writer.writerow([trial_count] + pattern + ['FAILED'])
                    results_file.flush()
                    continue
                    
                avs = doit(pattern, simtime=0.11, return_accel=False, rod = False, vis = False, control=False, use_ias=True)
                if avs is not None:
                    tests = tests + [avs]
                    results_writer.writerow([trial_count] + pattern + [a for a in tests[-1]])
                    results_file.flush()
                else:
                    tests = tests + [avs]
                    results_writer.writerow([trial_count] + pattern + ['FAILED'])
                    results_file.flush()

        print(tests)
        
def plot_vector_results(filename, check_muscs=[0]):

    num_muscles = 10
    # Basic trial vs force - peaks at biceps activation and no triceps, lower peaks at no biceps no triceps or both bis and tris
    forces = []
    names = []
    zero_x = 0
    zero_y = 0
    zero_z = 0
    with open(filename) as resfile:
        reader = csv.reader(resfile, delimiter=',', lineterminator='\n')
        i = 0
        for row in reader:
            if i == 0:
                zero_x = float(row[8])
                zero_y = float(row[9])
                zero_z = float(row[10])
            i += 1
            names = names + [int(row[0])]
            forces = forces + [float(row[8])-zero_x]
            
    fig, ax1 = plt.subplots()
    colorE = '#0000FF'
    ax1.set_xlabel('Trial', size=14)
    ax1.set_ylabel('x force', size=14)
    line1 = ax1.bar(names, [a if a > 0 else 0 for a in forces])
    #line1.set_label('Monte Carlo Simulation')
    ax1.tick_params(axis='y')
    
    fig.tight_layout()
    plt.show()
    
    # Remove the force produced by biceps and triceps
    forces = []
    names = []
    zero_x = 0
    zero_y = 0
    zero_z = 0
    musc_x = [0 for i in range(num_muscles)]
    with open(filename) as resfile:
        reader = csv.reader(resfile, delimiter=',', lineterminator='\n')
        i = 0
        for row in reader:
            if i == 0:
                zero_x = float(row[8])
                zero_y = float(row[9])
                zero_z = float(row[10])
            for j in range(num_muscles):
                if i == j+1:
                    musc_x[j] = float(row[8])-zero_x
            i += 1
            names = names + [int(row[0])]
            f = float(row[8])-zero_x
            for j in check_muscs:
                if row[j+1] == '1': #using musc
                    f -= musc_x[j]
            forces = forces + [f]
            
    fig, ax1 = plt.subplots()
    colorE = '#0000FF'
    ax1.set_xlabel('Trial', size=14)
    ax1.set_ylabel('x force', size=14)
    line1 = ax1.bar(names, [a if a > 0 else 0 for a in forces])
    #line1.set_label('Monte Carlo Simulation')
    ax1.tick_params(axis='y')
    
    fig.tight_layout()
    plt.show()
    
def plot_results(filename):
    forces = []
    names = []
    with open(filename) as resfile:
        reader = csv.reader(resfile, delimiter=',', lineterminator='\n')
        for row in reader:
            if (row[-1] == "FAILED"):
                continue
            names = names + [int(row[0])]
            forces = forces + [float(row[-1])]
            
    fig, ax1 = plt.subplots()
    colorE = '#0000FF'
    ax1.set_xlabel('Trial', size=14)
    ax1.set_ylabel('x force', size=14)
    line1 = ax1.bar(names, [a if a > 0 else 0 for a in forces])
    #line1.set_label('Monte Carlo Simulation')
    ax1.tick_params(axis='y')
    
    fig.tight_layout()
    plt.show()
    
    
def plot_possible_force_space(filename, pattern_mask=[1,1,1,1,1,1,1]):

    num_muscles = 10
    
    from matplotlib.colors import LightSource
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from mpl_toolkits.mplot3d.axes3d import Axes3D
    
    forces = []
    names = []
    zero_x = 0
    zero_y = 0
    zero_z = 0
    with open(filename) as resfile:
        reader = csv.reader(resfile, delimiter=',', lineterminator='\n')
        i = 0
        for row in reader:
            if i == 0:
                zero_x = float(row[11])
                zero_y = float(row[13])
                zero_z = float(row[12])
                
            if i <= num_muscles+1:
                print([row[0], -float(row[11])+zero_x,-float(row[13])+zero_y,-float(row[12])+zero_z])
                
            i += 1
            ignore = False
            for j in range(num_muscles):
                if pattern_mask[j] == 0 and int(row[j+1]) == 1:
                    ignore = True
                    break
            if ignore:
                continue
            names = names + [int(row[0])]
            forces = forces + [[-float(row[11]),-float(row[13]),-float(row[12])]]

    fig = plt.figure()
    ax = Axes3D(fig, computed_zorder=False, azim=66, elev=12)
    ax.grid(True)

    ax.set_xlabel('Forward Force (N)', size=18, y=20.5)

    ax.set_zlabel('Upward Force (N)', size=18, y=20.5)

    ax.set_ylabel('Lateral Force (N)', size=18, y=20.5)
    
    forces = np.array(forces)
    
    from scipy.spatial import ConvexHull
    
    hull = ConvexHull(forces)
    for simplex in hull.simplices:
        simps = [[forces[simplex[0]], forces[simplex[1]], forces[simplex[2]]]]
        ax.add_collection3d(Poly3DCollection(simps, facecolor='b', edgecolor='#333388', ls='-', linewidth=1, alpha=0.2, zorder=1))


    ax.scatter(forces[:,0], forces[:,1], forces[:,2], zorder=0)
    
    ax.scatter([0], [0], [0], zorder=0, color='#FF0000')
    
    ax.invert_yaxis()
    
    ps = [[200,0,0],[0,0,0]]
    ps = np.array(ps)
    ax.plot(ps[:,0], ps[:,1], ps[:,2], ls='-', color='#FF0000', zorder=0)
    ps = [[75,0,0],[0,0,0]]
    ps = np.array(ps)
    #ax.plot(ps[:,0], ps[:,1], ps[:,2], ls='-', color='#FF0000', zorder=0)
    
    plt.show()


#m.biceps_long
#m.brachioradialis
#m.triceps_long
#m.triceps_med
#m.delt_ant
#m.delt_med
#m.delt_post
#m.infraspinatus
#m.pec_maj
#m.lat
    
#file = "force_results_force_point_final_reduced_cort.csv"
#file = "force_results_force_point_with_infsp.csv"
#sweep(filename=file, start_trial=0, problem_trials=[] )
#print(doit([1,0,0,0,0,0,0,0,0,0], 0.11, False, False, True, False, False))
#plot_vector_results("force_results_vector.csv", check_muscs=[0,2,3,5])
#plot_possible_force_space(filename=file, pattern_mask=[1,0,1,1,1,0,1,0,1,0])

#valid combinations:
#1,0,1,0,0,1,1,0,0
#0,1,1,0,0,1,1,0,0
#0,1,1,0,0,1,0,0,0
#0,1,1,0,1,1,0,0,0
#0,1,1,0,1,0,0,0,0
#0,1,1,0,1,0,0,1,0
#0,1,1,0,1,1,0,1,0
#0,1,0,0,0,1,1,0,0
#1,1,0,0,0,1,1,0,0
#1,1,1,0,0,1,0,0,0

#BB, TL, LD, PD
#BRD, TL, AD
#BRD, Tl, AD, PD
#TM, PM, LAT - not important 

basic_simulation_example()