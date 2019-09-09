import glob
import os
import librosa
import numpy as np
import random


import torch
from d_vector.speech_embedder_net import SpeechEmbedder


import random
import time

# downloaded dataset path
#audio_path = glob.glob(os.path.dirname('./wav48/*/*.wav'))
audio_path = glob.glob(os.path.dirname('./development_set-2/*/wav/*.wav'))


def load_wav(utter_path,vad_time,utterances_spec):
    intervals=[]
    start_time=vad_time[0]
    end_time=vad_time[0]+1
    for i in range(1,len(vad_time)):
        if vad_time[i]==end_time:
            end_time=end_time+1
        else:
            intervals.append([start_time,end_time])
            start_time=vad_time[i]
            end_time=vad_time[i]+1
    intervals.append([start_time,end_time])
    utter, sr = librosa.core.load(utter_path, 16000)
      # save first and last 180 frames of spectrogram.
    for interval in intervals:
        if interval[1]==300:
            utter_part=utter[interval[0]*16000:interval[1]*16000-14000]
        else:
            utter_part=utter[interval[0]*16000:interval[1]*16000+2000]
        #utter_part=utter[interval[0]*16000-4000:interval[1]*16000]
        S = librosa.core.stft(y=utter_part, n_fft=512,
                            win_length=int(0.025 * 16000), hop_length=int(0.01 * 16000))
        S = np.abs(S) ** 2
        mel_basis = librosa.filters.mel(sr=16000, n_fft=512, n_mels=40)
        S = np.log10(np.dot(mel_basis, S) + 1e-6)           # log mel spectrogram of utterances
        frame_num=0
        n=1
        # step=0.25s,overlap=50%
        for j in range(0, int(S.shape[1]//12.5)-1):
            step_start=j*12+(n//2)
            utterances_spec.append(S[:,step_start:step_start+25])
            frame_num=frame_num+1
            n+=1
        #step=0.5s,overlap=50%
        #for j in range(0, int(S.shape[1]//25)-1):
        #    step_start=j*25
        #    utterances_spec.append(S[:,step_start:step_start+50])
        #    frame_num=frame_num+1
        #    n+=1
            
      
    return frame_num


def calculate_vad_accurary(time,id_path):
    result=[]
    vad_currect=0
    vad_wrong=0
    vad_total=0
    with open(id_path,'r') as f:
      for line in f:  
        result.append(line.strip('\n').split(':'))
    for i in range(len(result)):
        strat_time=result[i][0]
        current_label=result[i][1]
        if int(strat_time) not in time:
            vad_total+=1
            if current_label=='1' or current_label=='-1':
                vad_currect+=1
        else:
            if current_label=='1' or current_label=='-1':
                vad_wrong+=1
    vad_presicion=vad_currect/vad_total
    vad_recall=vad_currect/(vad_currect+vad_wrong)
    print('vad_presicion is %4f'%(vad_presicion))
    print('vad_recall is %4f'%(vad_recall))
    return vad_presicion,vad_recall


def make_dvector_test(train_sequence_name):
    device = torch.device("cpu")
    embedder_net = SpeechEmbedder().to(device)
    embedder_net.load_state_dict(torch.load('./d_vector/model/final_epoch_950_batch_id_244.model'))
    embedder_net.eval()
    utterances_spec = np.load(os.path.join('./uis_rnn_test', 'uis_rnn_test_utt.npy'), allow_pickle=True)
    train_sequence=[]
    for i in range(utterances_spec.shape[0]):
        utterances_spec_split=utterances_spec[i]
        utterances_spec_split=np.array(utterances_spec_split)
        print("utterances_spec.size",utterances_spec_split.shape)
        utterances_spec_split = torch.tensor(np.transpose(utterances_spec_split, axes=(0,2,1)))
        utterances_spec_split = utterances_spec_split.to(device)
        embeddings = embedder_net(utterances_spec_split)
        embeddings_numpy=embeddings.cpu().detach().numpy()
        train_sequence.append(embeddings_numpy)
    
    #train_sequence=np.concatenate(train_sequence, axis=0)
    #print("train_sequence.size",train_sequence.shape)
    np.save(os.path.join('./uis_rnn_test', train_sequence_name), train_sequence)
        

def create_dataset_real(vad_pathname,utter_path,id_path):
    os.makedirs('./uis_rnn_test', exist_ok=True)    # make folder to save test file
    #vad=np.load(vad_pathname)
    vad=vad_pathname
    time=[]
    for i in range(len(vad)):
        start=vad[i][0]
        time.append(start)
    
    calculate_vad_accurary(time,id_path)

    utt_id=0            #front label of cluster_id
    utterances_spec = []
    train_cluster_id=[]
    split_utterances_spec=[]
    split_train_cluster_id=[]
    utter_path = utter_path        # path of each utterance
    cluster_id=load_wav(utter_path,time,split_utterances_spec)
    train_cluster_id.append(cluster_id)
    utterances_spec.append(split_utterances_spec)
    #utterances_spec = np.stack(utterances_spec, axis=2)
    utterances_spec = np.array(utterances_spec)
    print(utterances_spec.shape)
    print("length of utt",len(utterances_spec))
    np.save(os.path.join('./uis_rnn_test', "uis_rnn_test_utt.npy"), utterances_spec)
    #np.save(os.path.join('./uis_rnn_test', "test_cluster_id_utt.npy"), train_cluster_id)





if __name__ == "__main__":
#    create_dataset()
    vad_pathname='vad_test1_3.npy'
    utter_path='./test_1.wav'
    train_sequence_name="0.25_second_test_sequence_utt1_3.npy"
    id_path='test_1.txt'
    
    create_dataset_real(vad_pathname,utter_path,id_path)
#    create_testdataset()
#    make_dvector()
    make_dvector_test(train_sequence_name)