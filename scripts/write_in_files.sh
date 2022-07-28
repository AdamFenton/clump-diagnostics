# /usr/bin/bash
# ----------------------------------#
# write_in_files.sh                 #
# ----------------------------------#
# Reads the lines in the
# run_parameters.dat file (this
# is created by the generate_
# production_run_table.py script) and
# takes a template disc.in file and
# creates one for each of the runs
# entering the values for EOS params
# ----------------------------------#
# Author: Adam Fenton
# Date:   20220728
# ----------------------------------#
counter=0
while read -r line; do
    counter=$((counter+1))
    stringarray=($line)
    new_rc1=${stringarray[0]}
    new_rc2=${stringarray[1]}
    new_rc3=${stringarray[2]}

    new_g1=${stringarray[3]}
    new_g2=${stringarray[4]}
    new_g3=${stringarray[5]}


    rhocrit1=`grep -w rhocrit1 disc_template.in | awk '{print $3}'`;sed -i '' "s|$rhocrit1|$new_rc1|g" disc_template.in;
    rhocrit2=`grep -w rhocrit2 disc_template.in | awk '{print $3}'`;sed -i '' "s|$rhocrit2|$new_rc2|g" disc_template.in;
    rhocrit3=`grep -w rhocrit3 disc_template.in | awk '{print $3}'`;sed -i '' "s|$rhocrit2|$new_rc3|g" disc_template.in;


    gamma1=`grep -w gamma1 disc_template.in | awk '{print $3}'`;sed -i '' "s|$gamma1|$new_g1|g" disc_template.in;
    gamma2=`grep -w gamma2 disc_template.in | awk '{print $3}'`;sed -i '' "s|$gamma2|$new_g2|g" disc_template.in;
    gamma3=`grep -w gamma3 disc_template.in | awk '{print $3}'`;sed -i '' "s|$gamma3|$new_g3|g" disc_template.in;

    run_dir="run_"`printf %03d $counter`
    cp disc_template.in $run_dir/disc.in
    echo $run_dir
done < run_parameters.dat
