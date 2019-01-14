import numpy as np
from random import randint
import math

import gym
env = gym.make('Blackjack-v0')

def sigmoid(x):
  return 1/(1 + math.exp(-x))

def dealNewCard():
  return randint(1,10)

class RNN:
  HIT = 1
  STAND = 0
  def __init__(self):
    self.inputToH1Weights = np.ones((3,2))
    self.H1ToH2Weights = np.ones((2,2))
    self.rnnOutpWs = np.ones((2,1))
    self.bWs = np.array([[1],[1]])

    self.rnnH2Activs = np.array([0,0])

  def weight(self,l, a,b):
    return l[a][b]

  def feedforward(self, inp, prevState):

    ps,ds,_ = inp

    self.inputActs = np.array([ps,ds,_])

    a11 = ((self.inputToH1Weights[0][0] * ps) + dealNewCard()) / (self.inputToH1Weights[0][1] * ds)
    a12 = (self.inputToH1Weights[1][0] * ps) / (self.inputToH1Weights[1][1] * ds)

    if (prevState == 1):
      a11 += 1
    else:
      a12 += 1

    a11s = sigmoid(a11)
    a12s = sigmoid(a12)


    self.rnnH1Acts = np.array([a11, a12])

    a21 = (self.H1ToH2Weights[0][0] * a11s) / (self.H1ToH2Weights[0][1] * a12s)
    a21s = sigmoid(a21)
    a22 = (self.H1ToH2Weights[1][0] * a12s) / (self.H1ToH2Weights[1][1] * a11s)
    a22s = sigmoid(a22)

    self.H2Acts = np.array([a21, a22])

    lessThan21 = a21s * self.rnnOutpWs[0]
    bust = a22s * self.rnnOutpWs[1]

    res,=np.array([a21s, a22s]).dot(self.rnnOutpWs)
    self.outActs = np.array([res])

    if lessThan21 > bust:
      return RNN.STAND
    else:
      return RNN.HIT


    # sig_res = sigmoid(res)
    #
    # if sig_res < 0.5:
    #   return RNN.HIT
    # else:
    #   return RNN.STAND
    
    # hit,stand=inp.dot(self.inputToH1Weights)
    #
    # hitratio = (ps + dealNewCard()) / ds
    # standratio = ds/ps
    #
    # (hit1,stnd1)=np.array([hitratio, standratio]).dot(self.H1ToH2Weights)
    #
    # lessThan21ratio = (hitratio * self.H1ToH2Weights[0][0])/(standratio * self.H1ToH2Weights[0][1])
    # bustratio = (standratio * self.H1ToH2Weights[1][0])/(hitratio * self.H1ToH2Weights[1][1])
    #
    # self.rnnH2Activs = np.array([hit1,stnd1])
    #
    # res,=np.array([lessThan21ratio, bustratio]).dot(self.rnnOutpWs)
    #
    # sigmVal = (sigmoid( res ))
    #
    # if sigmVal > 0.5:
    #   return RNN.HIT
    # else:
    #   return RNN.STAND

  def backprop(self, y_out, target):
    err = 0.5 * ((target - y_out) ** 2)
    
    # update h2 to y_out weights

    delta41 = (y^t) * sigmoid(self.outActs[0]) * (1-sigmoid(self.outActs[0]))

    self.rnnOutpWs[0] -= (target ^ y_out) * (sigmoid(self.outActs[0]) * (1-sigmoid(self.outActs[0]))) * self.H2Acts[0]
    self.rnnOutpWs[1] -= (target ^ y_out) * (sigmoid(self.outActs[0]) * (1-sigmoid(self.outActs[0]))) * self.H2Acts[1]

    self.H1ToH2Weights[0][0] -= sigmoid(self.H2Acts[0]) * (1-sigmoid(self.H2Acts[0])) * self.rnnOutpWs[0] * delta41 * sigmoid(self.rnnH1Acts[0])
    self.H1ToH2Weights[0][1] -= sigmoid(self.H2Acts[1]) * (1-sigmoid(self.H2Acts[1])) * self.rnnOutpWs[1] * delta41 * sigmoid(self.rnnH1Acts[1])
    self.H1ToH2Weights[1][0] -= sigmoid(self.H2Acts[0]) * (1-sigmoid(self.H2Acts[0])) * self.rnnOutpWs[0] * delta41 * sigmoid(self.rnnH1Acts[0])
    self.H1ToH2Weights[1][1] -= sigmoid(self.H2Acts[1]) * (1-sigmoid(self.H2Acts[1])) * self.rnnOutpWs[1] * delta41 * sigmoid(self.rnnH1Acts[1])

    delta31 =  sigmoid(self.H2Acts[0]) * (1-sigmoid(self.H2Acts[0])) * self.rnnOutpWs[0] * delta41
    delta32 =  sigmoid(self.H2Acts[1]) * (1-sigmoid(self.H2Acts[1])) * self.rnnOutpWs[1] * delta41

    self.inputToH1Weights[0][0] -= sigmoid(self.rnnH1Acts[0]) * (1-sigmoid(self.rnnH1Acts[0])) * (delta31 * self.H1ToH2Weights[0][0] + self.H1ToH2Weights[1][0] * delta32)
    self.inputToH1Weights[0][1] -= sigmoid(self.rnnH1Acts[1]) * (1-sigmoid(self.rnnH1Acts[1])) * (delta31 * self.H1ToH2Weights[0][1] + self.H1ToH2Weights[1][1] * delta32)
    self.inputToH1Weights[1][0] -= sigmoid(self.rnnH1Acts[0]) * (1-sigmoid(self.rnnH1Acts[0])) * (delta31 * self.H1ToH2Weights[0][0] + self.H1ToH2Weights[1][0] * delta32)
    self.inputToH1Weights[1][1] -= sigmoid(self.rnnH1Acts[1]) * (1-sigmoid(self.rnnH1Acts[1])) * (delta31 * self.H1ToH2Weights[0][1] + self.H1ToH2Weights[1][1] * delta32)



agent=RNN()

def game(agent):
  psums = dealNewCard(), dealNewCard()
  
  psum = sum(psums)
  dsum = dealNewCard()
  isAce = 1 if (1 in psums) else 0
  lastAction = None

  while psum < 21 or dsum <= 17:
    action = agent.feedforward(np.array([psum, dsum, isAce]), lastAction)
    lastAction = action
    if action == RNN.HIT:
      psum += dealNewCard()
      if psum > 21:
        return lastAction, lastAction ^ 1
      elif psum == 21:
        return lastAction, lastAction
    else:
      while dsum <= 17:
        dsum += dealNewCard()
      if dsum > 21:
        return lastAction, lastAction ^ 1
      elif dsum < psum:
        return lastAction, lastAction ^ 1
      else:
        return lastAction, lastAction

# print(agent.H1ToH2Weights)
print(agent.inputToH1Weights)
print(agent.H1ToH2Weights)
print(agent.rnnOutpWs)
totalErr=0
payout=0
for i in range(10000):
  y,t=game(agent)
  if y != t:
    totalErr += 0.5 * (y^t) ** 2
    payout -= 1
  else:
    payout += 1
  print(("HIT" if y==1 else "STAND","HIT" if t==1 else "STAND"), totalErr)
  agent.backprop(y,t)
print(payout/10000.0)
print()
print(agent.inputToH1Weights)
print(agent.H1ToH2Weights)
print(agent.rnnOutpWs)
