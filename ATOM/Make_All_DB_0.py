import pandas as pd
import os
import glob
import multiprocessing
import time
from multiprocessing import Manager


class worker_step1(multiprocessing.Process):
    def __init__(self, min_r, max_r, all_file_path, mem):
        multiprocessing.Process.__init__(self)
        self.min_r = min_r
        self.max_r = max_r
        self.target_file_list = all_file_path
        self.mem = mem
        self.E_B = 'BOC'

    def run(self):
        for _ in self.target_file_list[self.min_r: self.max_r]:
            self.one_file_load(_)

    def one_file_load(self, one_file_path):
        _, ac_type, nub = one_file_path.split('/')
        if self.E_B in ac_type:

            with open(one_file_path, 'r') as f:
                data = f.read().split('\n')

                # 컬럼 제작
                col_line = [out for out in data[0].split(' ') if out != '']

                # 단일 파일에서 데이터 분할
                DB = []
                for one_data in data[3:]:
                    one_line = one_data.split(' ')
                    DB.append([out for out in one_line if out != ''])

            self.done_file = pd.DataFrame(DB, columns=col_line)

            self.done_file['ac_type'] = ac_type

            if ac_type == f'{self.E_B}_MSLB_499':
                self.update_done_file([1, 0, 0])
            elif ac_type == f'{self.E_B}_MSLB_699':
                self.update_done_file([0, 1, 0])
            elif ac_type == f'{self.E_B}_SBLOCA':
                self.update_done_file([0, 0, 1])
            else:
                print('ERR')

            self.done_file.to_pickle(f'./mars_data/Out_db/{ac_type}_{nub[0:7]}.pkl')
            self.mem.append(self.done_file)
            print(f'Done - {ac_type}_{nub}')
        else:
            pass

    def update_done_file(self, control_list):
        # 특정 시나리오를 원 핫 인코딩 라벨링
        self.done_file[f'{self.E_B}_MSLB_499'] = control_list[0]
        self.done_file[f'{self.E_B}_MSLB_699'] = control_list[1]
        self.done_file[f'{self.E_B}_SBLOCA'] = control_list[2]


class main():
    def __init__(self):
        self.worker_list = []
        self.now_time = time.time()
        self.inter_val = 50
        self.mem = Manager().list()

        # 0. mother path 선정
        self.mother_path = os.getcwd()

        # 0. Out_db 청소
        file_list_in_Out_db = os.listdir(os.path.join(self.mother_path, f'mars_data/Out_db'))
        [os.remove(os.path.join(self.mother_path, f'mars_data/Out_db/{one_path}')) for one_path in file_list_in_Out_db]
        print('Out_db 청소')

        # 1. 폴더에서 plot 파일의 리스트 모두 읽음
        self.train_file_list = self.read_train_plot()

    def read_train_plot(self):
        file_list = []
        train_fold_path = os.listdir(os.path.join(self.mother_path, 'mars_data'))
        for _ in train_fold_path:
            one_fold_path = os.listdir(os.path.join(self.mother_path, f'mars_data/{_}'))
            for file in one_fold_path:
                if 'plot' in file:
                    file_list.append(os.path.join(self.mother_path, f'mars_data/{_}/{file}'))
        return file_list

    def run(self):
        # 전체 파일의 경로를 쓰레드에 할당
        for _ in range(0, len(self.train_file_list), self.inter_val):
            self.worker_list.append(worker_step1(min_r=_, max_r=_ + self.inter_val,
                                                 all_file_path=self.train_file_list,
                                                 mem=self.mem))

        for _ in self.worker_list:
            _.start()

        for _ in self.worker_list:
            _.join()

        self.tot_mem = []
        for _ in range(0, len(self.mem)):
            self.tot_mem.append(self.mem[_])

        all_db = pd.Series(self.tot_mem)

        print('rebuild')
        # all_db.to_pickle('./OUT_PUT/all_db.pkl')
        print(time.time() - self.now_time)

        # ==============================================
        from sklearn.preprocessing import MinMaxScaler
        sclaer = MinMaxScaler()
        for _ in range(len(all_db)):
            sclaer.fit(all_db[_].iloc[:, :-7])
            print(f'{_}/{len(all_db)}')
        print('Min-max Done')
        # ==============================================
        import pickle
        with open('./OUT_PUT/minmax.bin', 'wb') as f:
            pickle.dump(sclaer, f)
        print('Min-max Save')
        # ==============================================
        # min max 스케일러로 데이터 전체 전환
        for _ in range(len(all_db)):
            all_db[_].iloc[:, :-7] = sclaer.transform(all_db[_].iloc[:, :-7])
        all_db.to_pickle('./OUT_PUT/all_db_min_max.pkl')
        # ==============================================
        print(time.time() - self.now_time)
        print('DB 통합 완료')


if __name__ == '__main__':
    body = main()
    body.run()
