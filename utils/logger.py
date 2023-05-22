import os
import csv
from datetime import datetime
import wandb

class custom_logger():
    def __init__(self, output_path, name='log'):
        os.makedirs(output_path, exist_ok=True)
        now = datetime.now()
        logger_name = str(now.strftime("20%y_%h_%d_")) + name + ".txt"
        self.logger_path = os.path.join(output_path, logger_name)
        self.csv_path = os.path.join(output_path, logger_name.replace('.txt', '.csv'))
        # init logger file
        f = open(self.logger_path, "w+")
        f.write(self.get_local_time() + 'Start Logging \n')
        f.close()
        # init csv
        with open(self.csv_path, 'w') as f:
            writer = csv.writer(f)
            writer.writerow([self.get_local_time(), ])

    def get_local_time(self):
        now = datetime.now()
        return str(now.strftime("%y_%h_%d %H:%M:%S : "))

    def info(self, log_str):
        print(str(log_str))
        with open(self.logger_path, "a") as f:
            f.write(self.get_local_time() + str(log_str) + '\n')

    def raise_error(self, error):
        prototype = '************* Error: {} *************'.format(str(error))
        self.info(prototype)
        raise ValueError(str(error))

    def info_iter(self, epoch, batch, total_batch, info_dict, print_iter):
        if batch % print_iter != 0:
            pass
        else:
            acc_log = 'Epoch {:5d}, Batch {:6d}/{},'.format(epoch, batch, total_batch)
            for key, val in info_dict.items():
                acc_log += ' {}: {:9.3f},'.format(str(key), float(val))
            self.info(acc_log)

    def write_results(self, result_list):
        with open(self.csv_path, 'a') as f:
            writer = csv.writer(f)
            writer.writerow(result_list)

    def wandb_init(self, config , project, name):
        wandb.init(project=project, name=name, config=config)
        

    def wandb_log(self, log_dict):
        wandb.log(log_dict)

    def wandb_finish(self):
        wandb.finish()