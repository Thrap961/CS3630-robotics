from grid import *
from particle import Particle
from utils import *
from setting import *
import math

def motion_update(particles, odom):
    """ Particle filter motion update

        Arguments:
        particles -- input list of particle represents belief p(x_{t-1} | u_{t-1})
                before motion update
        odom -- odometry to move (dx, dy, dh) in *robot local frame*

        Returns: the list of particles represents belief \tilde{p}(x_{t} | u_{t})
                after motion update
    """
    motion_particles = []
    dx,dy,dh = odom
    for p in particles:
        dist = dx**2 + dy**2
        p.h += add_gaussian_noise(dh,ODOM_HEAD_SIGMA)
        p.h %= 360
        p.x += add_gaussian_noise(dist*math.cos(math.radians(dh)), ODOM_TRANS_SIGMA)
        p.y += add_gaussian_noise(dist*math.sin(math.radians(dh)), ODOM_TRANS_SIGMA)
        motion_particles.append(p)

    return motion_particles

# ------------------------------------------------------------------------
def measurement_update(particles, measured_marker_list, grid):
    """ Particle filter measurement update

        Arguments:
        particles -- input list of particle represents belief \tilde{p}(x_{t} | u_{t})
                before meansurement update (but after motion update)

        measured_marker_list -- robot detected marker list, each marker has format:
                measured_marker_list[i] = (rx, ry, rh)
                rx -- marker's relative X coordinate in robot's frame
                ry -- marker's relative Y coordinate in robot's frame
                rh -- marker's relative heading in robot's frame, in degree

                * Note that the robot can only see markers which is in its camera field of view,
                which is defined by ROBOT_CAMERA_FOV_DEG in setting.py
				* Note that the robot can see mutliple markers at once, and may not see any one

        grid -- grid world map, which contains the marker information,
                see grid.py and CozGrid for definition
                Can be used to evaluate particles

        Returns: the list of particles represents belief p(x_{t} | u_{t})
                after measurement update
    """
    measured_particles = []
    p = [1.0]*len(particles)

    # measurement update
    for i,par in enumerate(particles):
        # read markers based on current particle field of view
        pmarkers = p.read_markers(grid)

        if len(pmarkers) != 0:

            # find a best pairing marker in the pmarkers for each robot sensed marker
            ######################### I didn't consider the repeated paired pmarkers!!!! Does that matter? ###################### 
            for rm in measured_marker_list:
                # suppose paired marker is the first one
                pairedM = None
                minDistance = None
                diffAngle = None
                # loop through all the markers to find the closest marker
                for pm in pmarkers:
                    d = grid_distance(rm[0],rm[1],pm[0],pm[1])
                    if (not minDistance) || (minDistance > d):
                        minDistance = d
                        pairedM = pm
                        diffAngle = diff_heading_deg(rm[2],pm[2])

                power = - minDistance**2/(2*(MARKER_TRANS_SIGMA**2)) - diffAngle**2/(2*(MARKER_ROT_SIGMA**2))
                p[i] *= Math.exp(power)

        else:
            if len(measured_marker_list) != 0:
                p[i] = 0
        
        # if len(measured_marker_list), we will not update the weights (?is that okayy?)
    
    # normalize the weights
    p /= sum(p)
    # Todo: resample


    return measured_particles


