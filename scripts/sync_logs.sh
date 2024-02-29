#rsync -u -avz -r --exclude="*.npy" csshshi@gpuhome.comp.hkbu.edu.hk:/home/comp/csshshi/repositories/p2p-dl/logs/* ./logs/ 
rsync -u -avz -r --exclude="*.npy" --exclude="gpu*.log" shshi@158.182.78.12:/home/shshi/host143/repos/p2p-dl/logs/* ./logs/ 
rsync -u -avz -r --exclude="*.npy" --exclude="gpu*.log" shshi@158.182.78.12:/home/shshi/host144/repos/p2p-dl/logs/* ./logs/ 
#rsync -u -avz -r --exclude="*.npy" ./logs/* /home/comp/csshshi/backuplogs/
