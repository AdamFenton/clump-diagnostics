# Import pexpect module

import pexpect
import glob
import subprocess

# splash_titles_commands = 'ls -d -- *run* | sort -t. -k4 >> splash.titles'
complete_file_list = glob.glob("run1*")
complete_file_list = sorted(complete_file_list, key = lambda x: x.split('.')[3])
# subprocess.check_call(splash_titles_commands, shell=True)

def splash_interact_coodinate_plots(filename,PID,y_axis,x_axis,render,vector):
    ''' This function answers the splash command line prompts for a coodinate
        plot - specifically a coodinate plot (i.e. in x and y) because only these
        allow for particle rendering.
    '''
    savename = '%s.pdf' % filename
    child = pexpect.spawn('/net/europa2/work/afenton/splash_2021/bin/splash '+ filename)
    child.expect('') # Ask for option
    child.sendline('l3')
    child.expect('') # Ask for particleID
    child.sendline(PID)
    child.expect('') # Ask for xmin
    child.sendline('5')
    child.expect('') # Ask for xmax
    child.sendline('5')
    child.expect('') # Ask for ymin
    child.sendline('5')
    child.expect('') # Ask for ymax
    child.sendline('5')
    child.expect('') # Ask for zmin
    child.sendline('1.5')
    child.expect('') # Ask for zmax
    child.sendline('1.5')

    child.expect('') # Ask for option
    child.sendline('l5')
    child.expect('') # Ask for adaptive axes
    child.sendline('6')
    child.expect('') # Ask for adaptive axes
    child.sendline('1')
    child.expect('') # Ask for adaptive axes
    child.sendline('0')


    child.expect('') # Ask for option
    child.sendline('g2')
    child.expect('') # Ask for titles
    child.sendline('yes')
    child.expect('') # Ask for titles
    child.sendline('0.5')
    child.expect('') # Ask for titles
    child.sendline('1')
    child.expect('') # Ask for titles
    child.sendline('0.5')


    child.expect('') # Ask for option
    child.sendline('l1')
    child.expect('') # Ask for adaptive axes
    child.sendline('yes')
    child.expect('') # Ask for adaptive axes
    child.sendline('no')
    child.expect('') # Ask for adaptive axes
    child.sendline('no')




    child.expect('') # Ask for Y axis
    child.sendline(y_axis)
    child.expect('') # Ask for X axis
    child.sendline(x_axis)
    child.expect('') # Ask for render
    child.sendline(render)
    child.expect('') # Ask for vectory
    child.sendline(vector)
    child.expect('') # Ask for filename to save as
    child.sendline(savename)
    child.expect('') # Press q to quit
    child.sendline('q')

    child.interact()
for filename in complete_file_list:
    PID = filename.split(".")[2]
    subprocess.check_call('ls %s > splash.titles' % filename, shell=True)
    splash_interact_coodinate_plots(filename,PID,'2','1','6','0')
