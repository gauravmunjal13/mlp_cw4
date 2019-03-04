# Downloading data

* Download CIFAR-10 dataset from <https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz>, and extract to `data` directory

# Workflow on cluster
* Add commands to bottom of script, and run script via: `sbatch script_name.sh`. See example script for examples of commands.
* To view experiment results, go into the relevant directory in `save` and look for `<attack_type>_<attack_args>.txt`. 
For example, baseline (no attack) performance is `None_None.txt`, while FGSM with eps=1 is `fgsm_1.txt`.
* Command to transfer files from cluster -> local DICE: `scp s*******@mlp.inf.ed.ac.uk:/home/s*******/mlp_cw4/file_name /afs/inf/ed/ac/uk/user/s18/s*******/`
* One way to get files from local DICE is to upload them to google drive using an internet browser.