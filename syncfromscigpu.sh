#rsync -u -avz -r shaohuais@psglogin.nvidia.com:/home/shaohuais/repos/p2p-topk/logs/* ./logs/
#rsync -u -avz -r 15485625@scigpu10:/home/comp/15485625/repos/p2p-topk/logs/* ./logs/
rsync -u -avz -r 15485625@scigpu10:/home/comp/15485625/repos/p2p-iclr/logs/* ./logs/
#rsync -u -avz -r --include="**/dgx*.log" --exclude="*" shshi@158.182.9.141:/home/shshi/work/p2p-dl/logs/* ./logs/
