""" BNInception Runs """

""" Online Products our Method """
python trainModel.py --device=0 --repeat=3 --dataset=SOP --cfg=dml_large --loss=ccp/contrastive  


""" InShop our Method """
python trainModel.py --device=0 --repeat=3 --dataset=InShop --cfg=dml_large --loss=ccp/contrastive


""" Cars our Method """
python trainModel.py --device=0 --repeat=3 --dataset=Cars196 --cfg=dml_small --loss=ccp/contrastive

""" Cub our Method """
python trainModel.py --device=0 --repeat=3 --dataset=CUB200_2011 --cfg=dml_small --loss=ccp/contrastive
