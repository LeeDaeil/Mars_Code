import multiprocessing
from multiprocessing import Manager
import pickle
import pandas as pd
import numpy as np
import time
import math
import os


class Worker(multiprocessing.Process):
    def __init__(self, nub, mem_input, mem_output, divided_tot_db, time_leg, data_type, save_one_time):
        multiprocessing.Process.__init__(self)
        self.nub = nub
        self.mem_input = mem_input
        self.mem_output = mem_output
        self.temp_input = []
        self.temp_output = []
        # self.sel_colums = pd.read_csv('Para.csv')  # 선정된 파라메터 가져오기
        self.divided_tot_db = divided_tot_db       # 상위 수준에서 분할된 데이터 베이스
        self.time_leg = time_leg
        self.data_type = data_type

        self.save_para_name_one_time = save_one_time

    def run(self):
        i = 0
        for senario in self.divided_tot_db:
            ## 변수를 어떻게 선택 할 지 생각하는 부분 ------------------------------------------------
            # 진단 용 데이터베이스 제작
            if self.data_type == 'Dig':
                # 입력 데이터 베이스 제작 # pass
                # one_input = senario.iloc[:, 1:-7] #[self.sel_colums[(self.sel_colums['S_COM'] == 1) | (self.sel_colums['S_COND'] == 1)]['CNS']]
                one_input = senario.iloc[:, 1:-4] #[self.sel_colums[(self.sel_colums['S_COM'] == 1) | (self.sel_colums['S_COND'] == 1)]['CNS']]
                # 출력 데이터 베이스 제작
                # : 이 부분은 단순히 사전에 비정상 상태가 라벨링된 값을 사용한다.
                # one_output = senario.iloc[:, -6:]
                one_output = senario.iloc[:, -3:]
                for _ in range(0, len(one_input) - self.time_leg):
                    self.temp_input.append(one_input.iloc[_:_ + self.time_leg, :].to_numpy())
                    self.temp_output.append(one_output.iloc[_ + self.time_leg: _ + self.time_leg + 1, :].to_numpy()[0])
                print(f'{self.name} - {i} / {len(self.divided_tot_db)}')
                i += 1
            # 컨트롤 용 데이터베이스 제작
            else:
                print('구현되지 않은 기능')
                pass
            # elif self.data_type == 'Cont':
            #     # 입력 데이터 베이스 제작
            #     Temp_time = time.time()
            #     one_input = senario[self.sel_colums[self.sel_colums['CONT_IN'] == 1]['CNS']]
            #     # 출력 데이터 베이스 제작
            #     # : 제어의 경우 해당 DB의 변수를 고려하여 추가한다.
            #     self.one_output = senario
            #     self.CONT_output_change()
            #
            #     # 저장된 파라메터를 보고 싶으면 =========
            #     if not self.save_para_name_one_time['one']:
            #         print(f'{self.name}에서 파라메터 이름 저장')
            #         index_para_list = self.sel_colums.set_index('CNS')
            #
            #         out_para_reproduct = []
            #         for out_put_keys in self.one_output.columns:
            #             out_para_reproduct.append(out_put_keys.split('_')[0])
            #         para_list_out = pd.DataFrame(index_para_list.loc[out_para_reproduct]['DS'])
            #         para_list_out['OG'] = self.one_output.columns
            #         para_list_in = pd.DataFrame(index_para_list.loc[one_input.columns]['DS'])
            #         para_list_out.to_csv('output_parameters.csv')
            #         para_list_in.to_csv('input_parameters.csv')
            #         self.save_para_name_one_time['one'] = True
            #
            #     # 데이터 타임 길이만큼 자르기 =========
            #     for _ in range(0, len(one_input) - self.time_leg):
            #         self.temp_input.append(one_input.iloc[_:_ + self.time_leg, :].to_numpy())
            #         self.temp_output.append(self.one_output.iloc[_ + self.time_leg: _ + self.time_leg + 1, :].to_numpy()[0])
            #     with open(f'./Ttherd_out/{self.name}.txt', 'a') as f:
            #         se_name = senario['ab_nub'][0]
            #         f.write(f'{self.name}: {se_name} - {np.shape(self.temp_input)} - {np.shape(self.temp_output)}\n')
            #
            #     # print(f'End time: {time.time() - Temp_time}')
            #     print(f'{self.name} - {i} / {len(self.divided_tot_db)}')
            #     i += 1

        print(f'{self.name}: {np.shape(self.temp_input)} - {np.shape(self.temp_output)}')
        self.mem_output[self.nub] = np.array(self.temp_output)
        self.mem_input[self.nub] = np.array(self.temp_input)

    # def CONT_output_change(self):
    #     # CNS 제어 변수의 값을 변환하다. 아래 리스트로 변환된다.
    #
    #     temp_para_list = []
    #     self.cont_nub = 0   # self.cont_nub -> 현재 까지 축적된 변수 번호
    #     # 포지션이 있는 변수는 모두 해당된다.
    #     for rg_para in self.sel_colums[self.sel_colums['CONT_RG'] == 1]['CNS']:
    #         temp_para_list.append(self.para_rg(rg_para))
    #
    #     # 제어봉의 조작의 경우
    #     temp_para_list.append(self.para_rod_control())
    #
    #     # 자동 수동이 있는 변수는 모두 해당된다.
    #     for am_para in self.sel_colums[self.sel_colums['CONT_AM'] == 1]['CNS']:
    #         temp_para_list.append(self.para_am(am_para))
    #
    #     self.one_output = pd.concat(temp_para_list, axis=1)
    #
    # def para_rg(self, para_name=''):
    #     # self.cont_nub -> 현재 까지 축적된 변수 번호
    #     temp_one_output = pd.DataFrame(self.one_output[para_name], columns=[para_name])
    #     temp_one_output[f'{para_name}_1'] = [temp_one_output[para_name][0]] + temp_one_output[para_name][0:-1].to_list()
    #     temp_one_output[f'{para_name}_2'] = temp_one_output[f'{para_name}'] - temp_one_output[f'{para_name}_1']
    #     temp_one_output[f'{para_name}_{self.cont_nub}_D'] = temp_one_output[f'{para_name}_2'].apply(lambda x: 1 if x < 0 else 0)
    #     temp_one_output[f'{para_name}_{self.cont_nub}_S'] = temp_one_output[f'{para_name}_2'].apply(lambda x: 1 if x == 0 else 0)
    #     temp_one_output[f'{para_name}_{self.cont_nub}_U'] = temp_one_output[f'{para_name}_2'].apply(lambda x: 1 if x > 0 else 0)
    #     self.cont_nub += 1
    #     return temp_one_output.iloc[:, -3:]
    #
    # def para_am(self, para_name =''):
    #     # self.cont_nub -> 현재 까지 축적된 변수 번호
    #     temp_one_output = pd.DataFrame(self.one_output[para_name], columns=[para_name])
    #     temp_one_output[f'{para_name}_{self.cont_nub}_A'] = temp_one_output[f'{para_name}'].apply(lambda x: 1 if x == 0 else 0)
    #     temp_one_output[f'{para_name}_{self.cont_nub}_M'] = temp_one_output[f'{para_name}'].apply(lambda x: 1 if x != 0 else 0)
    #     self.cont_nub += 1
    #     return temp_one_output.iloc[:, -2:]
    #
    # def para_rod_control(self):
    #     # self.cont_nub -> 현재 까지 축적된 변수 번호
    #     Cont = pd.DataFrame()
    #     Cont['tot_rod_pos'] = self.one_output['KBCDO3']
    #     for iter_para in ['KBCDO4', 'KBCDO5', 'KBCDO6', 'KBCDO7', 'KBCDO8', 'KBCDO9', 'KBCDO10']:
    #         Cont['tot_rod_pos'] += self.one_output[iter_para]
    #     para_name = 'KSWO22'
    #     Cont[f'{para_name}_1'] = [Cont['tot_rod_pos'][0]] + Cont['tot_rod_pos'][0:-1].to_list()
    #     Cont[f'{para_name}_2'] = Cont[f'tot_rod_pos'] - Cont[f'{para_name}_1']
    #     Cont[f'{para_name}_{self.cont_nub}_D'] = Cont[f'{para_name}_2'].apply(lambda x: 1 if x < 0 else 0)
    #     Cont[f'{para_name}_{self.cont_nub}_S'] = Cont[f'{para_name}_2'].apply(lambda x: 1 if x == 0 else 0)
    #     Cont[f'{para_name}_{self.cont_nub}_U'] = Cont[f'{para_name}_2'].apply(lambda x: 1 if x > 0 else 0)
    #     self.cont_nub += 1
    #     return Cont.iloc[:, -3:]


class main_body():
    def __init__(self, data_type):
        '''

        :param data_type: Dig, Cont
        '''
        self.start_time = time.time()
        # CPU 특성
        self.worker_number = 5
        # 공유 데이터
        self.mem_list_input = Manager().dict({i: None for i in range(self.worker_number)})
        self.mem_list_output = Manager().dict({i: None for i in range(self.worker_number)})
        self.save_para_one_t = Manager().dict({'one': False})
        # DB 특성
        self.time_leg = 10
        self.data_type = data_type
        self.tot_db = pd.read_pickle('./OUT_PUT/all_db_min_max.pkl')
        self.interval = math.ceil(len(self.tot_db) / self.worker_number)  # Worker의 숫자에 따라 간격 정의

        print('전체 데이터 로드 완료')
        self.tot_db = self.tot_db.sample(frac=1).reset_index(drop=True)
        print('전체 데이터 셔플 및 데이터 축적 시작')

        # 셔플로 생성된 데이터 리스트의 길이 및 예측 가능한 시나리오의 위치 분석
        see_nub, start_point = 0, 0
        with open('./OUT_PUT/db_history.txt', 'a') as f:
            f.write('\n')
        os.remove('./OUT_PUT/db_history.txt')
        for seen in self.tot_db:
            tot_line_seen = len(seen) - self.time_leg # Time leg를 꼭 고려해야함
            seen_type = seen['ac_type'][0]
            with open('./OUT_PUT/db_history.txt', 'a') as f:
                f.write(f'{see_nub}/{seen_type} - {start_point} ~ {start_point + tot_line_seen}\n')
            start_point += tot_line_seen
            see_nub += 1



        # Worker add
        self.worker_list = []
        for _ in range(0, self.worker_number):
            self.worker_list.append(Worker(
                nub=_,
                mem_input=self.mem_list_input,
                mem_output=self.mem_list_output,
                divided_tot_db=self.tot_db[_ * self.interval:(_ + 1) * self.interval],
                time_leg=self.time_leg,
                data_type=self.data_type, save_one_time=self.save_para_one_t)
            )
        print('All thread start')

    def run(self):
        job_list = []
        for _ in self.worker_list:
            _.start()
            job_list.append(_)

        for _ in job_list:
            _.join()

        # 축적된 데이터의 프로퍼티
        print('=' * 20)
        print(f'End time: {time.time() - self.start_time}')
        for woker_db_list in range(len(self.mem_list_output)):
            print(woker_db_list, np.shape(self.mem_list_output[woker_db_list]))
        print('=' * 20)

        # 데이터 병합 시작
        self.final_output = np.concatenate(self.mem_list_output, axis=0)
        self.final_input = np.concatenate(self.mem_list_input, axis=0)

        print('=' * 20)
        print(f'End time: {time.time() - self.start_time}')
        print(f'{self.final_output.shape} - {self.final_input.shape}')
        print('=' * 20)

        # 데이터 저장
        with open('./OUT_PUT/all_db_min_max_train_DB.bin', 'wb') as f:
            train_db = [self.final_input, self.final_output]
            pickle.dump(train_db, f)

        print(f'End time: {time.time() - self.start_time}')


if __name__ == '__main__':
    body = main_body(data_type='Dig')
    body.run()
