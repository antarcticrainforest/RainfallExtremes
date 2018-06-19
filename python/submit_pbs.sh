#!/bin/bash

echoerr() {
  echo "$@" 1>&2
  exit 257
}


# Construnct the directories
mkdir -p ~/.ens_jobs
rm -rf ~/.ens_jobs/pbs_submit.sh 2> /dev/null

#for m in $(module list 2>&1 |grep -iv currently|awk '{print $NF}'|grep -v found|uniq);do
#  mod=$(echo $m|rev | cut -d"/" -f2-  | rev)
#  modules=$(echo -n "${modules}module load $m\n")
#done
jobdir=~/.ens_jobs
vars=( 'ph' 'pg' 'pf' 'pe' 'pd' 'pc' 'pb' 'pa' 'pvera' 'pverb' 'pverc' )
for v in ${vars[*]};do
workdir=$(readlink -f $(dirname $0))
modules=$(echo -e $modules)
cat << EOF >> ~/.ens_jobs/pbs_submit-${v}.sh
#!/bin/bash
# set project
#PBS -P w40
# set stdout/stderr location
#PBS -o ${jobdir%/}/${v}.out
#PBS -e ${jobdir%/}/${v}.err
#PBS -l wd
# email options (abort,beg,end)
#PBS -m ae
#PBS -M martin.bergemann@monash.edu
# set name of job
#PBS -N ensemble-${v}
#PBS -q expressbw
#PBS -l walltime=12:00:00
#PBS -l mem=14GB
#PBS -l ncpus=4
module use /g/data3/hh5/public/modules
module use ~access/modules
module load pbs
module load conda/analysis3
export PYTHONWARNINGS="ignore"
cd ${workdir}
python ${workdir%/}/Ensemble.py $@ ${v}
EOF

chmod +x ~/.ens_jobs/pbs_submit-${v}.sh
echo submitting ~/.ens_jobs/pbs_submit-${v}.sh via qsub
qsub ~/.ens_jobs/pbs_submit-${v}.sh
done
