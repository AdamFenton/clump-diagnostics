import pexpect
import glob

complete_file_list = glob.glob("run1*")


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
    child.sendline('10')
    child.expect('') # Ask for xmax
    child.sendline('10')
    child.expect('') # Ask for ymin
    child.sendline('10')
    child.expect('') # Ask for ymax
    child.sendline('10')
    child.expect('') # Ask for zmin
    child.sendline('10')
    child.expect('') # Ask for zmax
    child.sendline('10')

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
    splash_interact_coodinate_plots(filename,PID,'2','1','6','0')
