cd /home/jgross/kinodata-docked-rescore
export WANDB_API_KEY=$(cat wandb_api_key)
for ID in 0 1 2 3 4
do	
	python3 train_dti_baseline.py --data_split "data/splits/random/seed_$ID.csv"
	python3 train_dti_baseline.py --data_split "data/splits/scaffold/seed_$ID.csv"
	python3 train_dti_baseline.py --data_split "data/splits/pocket/seed_$ID.csv"
done
