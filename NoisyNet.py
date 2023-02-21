"""

*reference : https://liger82.github.io/rl/rl/2021/09/01/DeepRLHandsOn-ch08-DQN_Extensions.html
*reference : https://github.com/thomashirtz/noisy-networks
*reference : https://github.com/thomashirtz/q-learning-algorithms/blob/master/noisy_networks/noisy_deep_q_learning.py

Noisy Network를 바로 적용하기 편하게 Module로 만들어놓자.

장점 :

    상당한 exploration이 필요할 때 요긴하다. 주로 NN base approach에서 성능이 좋을 것 같다. (Continuous space)

단점 :

    Computational cost가 올라간다. (Especially in case of non-factorized gaussian noise)
    Noisy gradient가 있기 때문에 agent가 조금 덜 안정적일 수 있다.
    최적의 Sigma, uniform bound 등의 설정이 조금 모호해서 이것 또한 tuning이 필요할 수 있다.

"""
import sys
import torch
import torch.nn as nn

"""
# nn.Parameter -> nn.Module 클래스의 attribute로 할당하면 자동으로 학습 파라미터 list에 추가된다.
# self.register_buffer로 layer를 등록하면?

    - optimizer가 업데이트 하지 않는다. 
    - 즉, End-to-end로 네트워크를 구성할 때 중간에 업데이트 하지 않는 일단 layer를 넣을 때 사용할 수 있다.  
    - GPU에서 작동
    - state_dict에는 저장이 된다. (값이 확인된다.)                               
"""

# Unfactorized gaussian noise -> gaussian distribution에서 noise를 weight : q x p 개 + bais : p개 뽑기
# Computational time이 크지만 일단 구현에 의미를 두자
# 나중에 Factorized gaussian noise로 만들어보자.

class NoisyLayer(nn.Module):
    # : , ->       그냥 주석의 역할
    def __init__(self, input_features : int, output_features : int, sigma = 0.017):
        super().__init__() # nn.Moudle 내의 constructor에 대해서는 따로 초기설정 건들지 않는다.

        self.sigma           = sigma
        self.input_features  = input_features
        self.output_features = output_features

        self.bound = (3/input_features) ** 0.5

        # trainable parameter 선언
        self.mu_bias    = nn.Parameter(torch.FloatTensor(output_features))
        self.sigma_bias = nn.Parameter(torch.FloatTensor(output_features))

        self.mu_weight    = nn.Parameter(torch.FloatTensor(output_features, input_features))
        self.sigma_weight = nn.Parameter(torch.FloatTensor(output_features, input_features))

        #self.epsilon_weight = None
        #self.epsilon_bias   = None

        self.register_buffer('epsilon_weight',  torch.FloatTensor((output_features, input_features)))
        self.register_buffer('epsilon_bias',    torch.FloatTensor((output_features, )))

        #self.register_buffer('epsilon_input',  torch.FloatTensor(input_features))
        #self.register_buffer('epsilon_output', torch.FloatTensor(output_features))

        self.parameter_initialization()            # instance 생성시 수행 -> 학습 파라미터 초기화
        self.sample_noise()                        # instance 생성시 수행 -> epsilon 값 생성

    def parameter_initialization(self): # Init trainable paramters (mu, sigma)
        # .fill(x) -> x로 채우기
        # .data    -> in place 연산
        self.mu_weight.data.uniform_(-self.bound, self.bound)
        self.mu_bias.data.uniform_(-self.bound, self.bound)
        self.sigma_bias.data.fill_(self.sigma)
        self.sigma_weight.data.fill_(self.sigma)

    def forward(self,  x : torch.Tensor, sample_noise : bool = True ):

        if not self.training: # noise 없이 forward pass
            return nn.functional.linear(x, weight = self.mu_weight, bias = self.mu_bias)

        # Forward pass시 매번 noise sampling해서 weight, bias에 섞는다.
        if sample_noise:
            self.sample_noise() # epsilon weight , epsilon bias noise 생성

        # Noisy weight / bias
        noisy_weight = self.mu_weight + (self.sigma_weight * self.epsilon_weight)
        noisy_bias   = self.mu_bias   + (self.sigma_bias   * self.epsilon_bias)

        return nn.functional.linear(x, weight = noisy_weight, bias = noisy_bias) # nn.Module의 weight, bias를 내껄로 설정

    def sample_noise(self):
        self.epsilon_weight = self.get_noise_tensor((self.output_features, self.input_features))  # q x p
        self.epsilon_bias   = self.get_noise_tensor((self.output_features,))                      # p x 1

    def get_noise_tensor(self, features : int):
        # .uniform_(a,b) : a와 b 사이의 균등 분포에서 tensor를 뽑뽑 -> unpacking에 주의
        noise = torch.FloatTensor(*features).uniform_(-self.bound, self.bound).to(self.mu_bias.device)
        return torch.sign(noise) * torch.sqrt(torch.abs(noise))


