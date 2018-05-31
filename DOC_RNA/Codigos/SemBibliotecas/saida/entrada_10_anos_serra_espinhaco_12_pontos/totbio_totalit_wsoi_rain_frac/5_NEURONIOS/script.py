#!/usr/bin/env python
# -*- coding: utf-8 -*-
# XOR Multilayer Perceptron usando BackPropagation
#
# Copyright (c) 2011, Antonio Rodrigo
# All rights reserved.
# Baseado no algoritmo de Neil Schemenauer <nas@arctrix.com>
import random

import math
import random
import numpy
import os
import pickle
from shutil import copyfile

random.seed(0)

N_NEURONIOS=5
frequencia_teste=100
variaveis_entrada="totbio totalit wsoi rain frac"
dirroot='/home/leticia/CCST/REDES_INLAND_2018/INLAND_BOX/dados/inland/inland_rna/data/RNA_FIRE/'
name_exp='entrada_10_anos_serra_espinhaco_12_pontos'
path_input=dirroot+name_exp

path_output='saida/'+name_exp+'/totbio_totalit_wsoi_rain_frac/'+str(N_NEURONIOS)+'_NEURONIOS/'
if not os.path.exists(path_output):
    os.makedirs(path_output)
copyfile(path_input, path_output+'/input')
copyfile('mpl.py', path_output+'/script.py')


# corrigir o erro = TERM environment variable not set.
# os.environ["TERM"] = 'xterm'

# gera numeros aleatorios obedecendo a regra:  a <= rand < b
def criar_linha():
    print "-" * 80


def rand(a, b):
    return (b - a) * random.random() + a


# nossa funcao sigmoide - gera graficos em forma de S
# funcao tangente hiperbolica
def funcao_ativacao_tang_hip(x):
    return math.tanh(x)


# derivada da tangente hiperbolica
def derivada_funcao_ativacao(x):
    t = funcao_ativacao_tang_hip(x)
    return 1 - t ** 2


# Normal logistic function.
# saída em [0, 1].
def funcao_ativacao_log(x):
    return 1 / (1 + math.exp(-x))


# derivada da função
def derivada_funcao_ativacao_log(x):
    return log(x) * (1 - log(x))


# Logistic function with output in [-1, 1].
def funcao_ativacao_log2(x):
    return 1 - 2 * log(x)


# derivada da função
def derivada_funcao_ativacao_log2(x):
    return -2 * log(x) * (1 - log(x))



def convertMes(out):
    #e=(float(temp[i])-min[i])/(max[i] - min[i])
    diff=12-1
    diff2=(out*diff)+1
    return diff2

def convertSaida(out):
    #e=(float(temp[i])-min[i])/(max[i] - min[i])
    diff=RedeNeural.max_output-RedeNeural.min_output
    diff2=(out*diff)+RedeNeural.min_output
    return diff2

class RedeNeural:
    max_output=0
    min_output=0
    def __init__(self, nos_entrada, nos_ocultos, nos_saida):
        # camada de entrada.data
        self.nos_entrada = nos_entrada + 1  # +1 por causa do no do bias
        # camada oculta
        self.nos_ocultos = nos_ocultos
        # camada de saida
        self.nos_saida = nos_saida
        # quantidade maxima de interacoes
        self.max_interacoes = 10000000
        # taxa de aprendizado
        self.taxa_aprendizado = 0.0005
        # momentum Normalmente eh ajustada entre 0.5 e 0.9
        self.momentum = 0.05

        self.indice_interacao=0

        # activations for nodes
        # cria uma matriz, preenchida com uns, de uma linha pela quantidade de nos
        self.ativacao_entrada = numpy.ones(self.nos_entrada)
        self.ativacao_ocultos = numpy.ones(self.nos_ocultos)
        self.ativacao_saida = numpy.ones(self.nos_saida)

        # contém os resultados das ativações de saída
        self.resultados_ativacao_saida = numpy.ones(self.nos_saida)

        # criar a matriz de pesos, preenchidas com zeros
        self.wi = numpy.zeros((self.nos_entrada, self.nos_ocultos))
        self.wo = numpy.zeros((self.nos_ocultos, self.nos_saida))

        # adicionar os valores dos pesos
        # vetor de pesos da camada de entrada.data - intermediaria
        for i in range(self.nos_entrada):
            for j in range(self.nos_ocultos):
                self.wi[i][j] = rand(-0.2, 0.2)

        # vetor de pesos da camada intermediaria - saida
        for j in range(self.nos_ocultos):
            for k in range(self.nos_saida):
                self.wo[j][k] = rand(-2.0, 2.0)

        # last change in weights for momentum
        self.ci = numpy.zeros((self.nos_entrada, self.nos_ocultos))
        self.co = numpy.zeros((self.nos_ocultos, self.nos_saida))


    def fase_forward(self, entradas):
        # input activations: -1 por causa do bias
        for i in range(self.nos_entrada - 1):
            self.ativacao_entrada[i] = entradas[i]

        # calcula as ativacoes dos neuronios da camada escondida
        for j in range(self.nos_ocultos):
            soma = 0
            for i in range(self.nos_entrada):
                soma = soma + self.ativacao_entrada[i] * self.wi[i][j]
            self.ativacao_ocultos[j] = funcao_ativacao_tang_hip(soma)

        # calcula as ativacoes dos neuronios da camada de saida
        # Note que as saidas dos neuronios da camada oculta fazem o papel de entrada.data
        # para os neuronios da camada de saida.
        for j in range(self.nos_saida):
            soma = 0
            for i in range(self.nos_ocultos):
                soma = soma + self.ativacao_ocultos[i] * self.wo[i][j]
            self.ativacao_saida[j] = funcao_ativacao_tang_hip(soma)

        return self.ativacao_saida

    def fase_backward(self, saidas_desejadas):
        # calcular os gradientes locais dos neuronios da camada de saida
        output_deltas = numpy.zeros(self.nos_saida)
        erro = 0
        for i in range(self.nos_saida):
            erro = saidas_desejadas[i] - self.ativacao_saida[i]
            output_deltas[i] = derivada_funcao_ativacao(self.ativacao_saida[i]) * erro

        # calcular os gradientes locais dos neuronios da camada escondida
        hidden_deltas = numpy.zeros(self.nos_ocultos)
        for i in range(self.nos_ocultos):
            erro = 0
            for j in range(self.nos_saida):
                erro = erro + output_deltas[j] * self.wo[i][j]
            hidden_deltas[i] = derivada_funcao_ativacao(self.ativacao_ocultos[i]) * erro

        # a partir da ultima camada ate a camada de entrada.data
        # os nos da camada atual ajustam seus pesos de forma a reduzir seus erros
        for i in range(self.nos_ocultos):
            for j in range(self.nos_saida):
                change = output_deltas[j] * self.ativacao_ocultos[i]
                self.wo[i][j] = self.wo[i][j] + (self.taxa_aprendizado * change) + (self.momentum * self.co[i][j])
                self.co[i][j] = change

        # atualizar os pesos da primeira camada
        for i in range(self.nos_entrada):
            for j in range(self.nos_ocultos):
                change = hidden_deltas[j] * self.ativacao_entrada[i]
                self.wi[i][j] = self.wi[i][j] + (self.taxa_aprendizado * change) + (self.momentum * self.ci[i][j])
                self.ci[i][j] = change

        # calcula erro
        erro = 0
        for i in range(len(saidas_desejadas)):
            erro = erro + 0.5 * (saidas_desejadas[i] - self.ativacao_saida[i]) ** 2
        return erro

    def test(self, entradas_saidas):
        acertos = 0
        print "Mês\tobservação\tsimulado\tdiff" 
        somadiff=0
        for p in entradas_saidas:
            array = self.fase_forward(p[1])
            saidaencontrada = array[0]
            #print convertSaida(p[2][0])
            if ( (convertSaida(p[2][0])*1.5)>=convertSaida(saidaencontrada)>=(convertSaida(p[2][0])*0.5)):
                acertos += 1
            print str(p[0][1])+"\t%06.2f" % convertSaida(p[2][0])+"\t\t%06.2f" % convertSaida(saidaencontrada)+"\t\t%08.2f" % (-100+((convertSaida(saidaencontrada)*100)/convertSaida(p[2][0])))+"%"
            somadiff+=abs(-100+((convertSaida(saidaencontrada)*100)/convertSaida(p[2][0])))


        print "Mês\tobservação\testimada\tdiff"  

        print  "\n>>>>>>>>>>>>>>>>>>>>\n Erro:\t%05.4f"%(somadiff/entradas_saidas.__len__())+'%\n>>>>>>>>>>>>>>>>>>>>\n'
        print  "\n>>>>>>>>>>>>>>>>>>>>\n Acertos:\t%05.4f"%(acertos*100/entradas_saidas.__len__())+'%\n>>>>>>>>>>>>>>>>>>>>\n'

        return somadiff/entradas_saidas.__len__()
        #for p in entradas_saidas:
        #    array = self.fase_forward(p[1])
        #    saidaencontrada= int (round( array[0]))
        #    msg="errou"
        #    if(saidaencontrada==p[2][0]):
        #        msg = "acertou"
        #    print("Entradas: " + str(p[1]) + ' - Saída encontrada/fase forward: ' + str(saidaencontrada)) + " "+msg

    def valida(self, entradas_saidas):
        acertos = 0
        for p in entradas_saidas:
            array = self.fase_forward(p[1])
            saidaencontrada = array[0]
            print p[1][0],convertSaida(p[2][0]),convertSaida(saidaencontrada)

    def valida2(self, entradas_saidas):
        acertos = 0
        validacao=[]
        for p in entradas_saidas:
            linha=[[],[],[]]
            array = self.fase_forward(p[1])
            saidaencontrada = array[0]
            linha[0]=p[0][1]
            linha[1]=convertSaida(p[2][0])
            linha[2]=convertSaida(saidaencontrada)
            validacao.append(linha)
        return validacao    

    def test2(self, input):
        array = self.fase_forward(input)
        saidaencontrada = array[0]
        return saidaencontrada

    def treinar(self, entradas_saidas, entradas_saidas_test, entradas_saidas_validacao):
        erro = 0
        err10 = 1000#diferença entre épocas
        while err10 > 0.000000009:
            errppp = 0
            for i in range(self.max_interacoes):
                
                erro = 0
                for p in entradas_saidas:
                    entradas = p[1]
                    saidas_desejadas = p[2]
                    self.fase_forward(entradas)
                    erro = erro + self.fase_backward(saidas_desejadas)

                #print "intera " + str(i) + " Erro = %2.10f" % erro + " Diferença: " + str(errppp - erro )
                if i % frequencia_teste == 0:
                    self.indice_interacao+=i

                    #self.test(entradas_saidas)
                    #print "intera " + str(i) + " >>>>>>>>>>>>>> Erro = %2.3f" % erro +" er "+ str( err10 - erro )
                    err10=erro
                    if erro <= 0.0005 or self.test(entradas_saidas_test)<5:
                        print 'parou - nº series: ' + str(i)
                        break

                    print variaveis_entrada
                    print "N_NEURONIOS:"+str(N_NEURONIOS)+" Época " + str(self.indice_interacao) + " >>>>>>>>>>>>>> Erro = %2.9f" % erro
                    binary_file = open(path_output+'_epoca_%.10d'%self.indice_interacao+'_rna.bin',mode='wb')
                    my_pickled_mary = pickle.dump(self, binary_file)
                    binary_file.close()
                    validacao=self.valida2(entradas_saidas_validacao)
                    plotResultado(self.indice_interacao,validacao,erro)
                    escreveFileErro(self.indice_interacao,erro)
    
                if errppp != erro:
                    errppp = erro

            if erro <= 0.0005 or self.test(entradas_saidas_test)<5:
              #  print 'parou - nº series: ' + str(i)
                break




        #for i in range(self.max_interacoes):
        #    erro = 0
        #    for p in entradas_saidas:
        #        entradas = p[1]
        #        saidas_desejadas = p[2]
        #        self.fase_forward(entradas)
        #        erro = erro + self.fase_backward(saidas_desejadas)
        #        print "Erro = %2.3f" % erro

        #    if i % 100 == 0:
        #        print "Erro = %2.3f" % erro
         #       if erro<=0.009:
         #           print 'parou - nº series: ' + str(i)
         #           break





def escreveFileErro(epoca,erro):
    fh = open(path_output+"erros.txt", 'a') 
    fh.write(' %.10d'%epoca+' %09.4f'%erro) 
    fh.close 



def plotResultado(indice,validacao,erro):
    import matplotlib.pyplot as plt
    import numpy as np

    #data_a = [[1,2,5], [5,7,2,2,5], [7,2,5]]
    #data_b = [[6,4,2], [1,2,5,3,2], [2,3,5,1]]

    data_a=[[],[],[],[],[],[],[],[],[],[],[],[]]
    data_b=[[],[],[],[],[],[],[],[],[],[],[],[]]

    texto=validacao
    for l in range(0,len(texto),1):
        temp = texto[l]
        mes=int(temp[0])-1
        observacao=temp[1]
        simulado=temp[2]
        data_a[mes].append(observacao)
        data_b[mes].append(simulado)

    ticks = ['JAN', 'FEV', 'MAR', 'ABR', 'MAI', 'JUN', 'JUL', 'AGO','SET','OUT','NOV','DEZ']

    def set_box_color(bp, color):
        plt.setp(bp['boxes'], color=color)
        plt.setp(bp['whiskers'], color=color)
        plt.setp(bp['caps'], color=color)
        plt.setp(bp['medians'], color='black')


    plt.figure()

    #bpl = plt.boxplot(data_a, positions=np.array(xrange(len(data_a)))*2.0-0.4, sym='', widths=0.6,showmeans=True,patch_artist=True)
    #bpr = plt.boxplot(data_b, positions=np.array(xrange(len(data_b)))*2.0+0.4, sym='', widths=0.6,showmeans=True,patch_artist=True)
    bpl = plt.boxplot(data_a, positions=np.array(xrange(len(data_a)))*2.0-0.4, sym='', widths=0.6,patch_artist=True)
    bpr = plt.boxplot(data_b, positions=np.array(xrange(len(data_b)))*2.0+0.4, sym='', widths=0.6,patch_artist=True)
   
    set_box_color(bpl, '#2C7BB6') # colors are from http://colorbrewer2.org/
    set_box_color(bpr, '#D7191C')

    # draw temporary red and blue lines and use them to create a legend
    plt.plot([], c='#2C7BB6', label='observacao')
    plt.plot([], c='#D7191C', label='simulado')
    plt.legend(loc='upper left')
    plt.title("Ocorrencia de fogo na Serra do Espinhaco entre 2002 e 2017")
    plt.ylabel("Numero de deteccoes de fogo")
    plt.xticks(xrange(0, len(ticks) * 2, 2), ticks)
    plt.xlim(-2, len(ticks)*2)
    #plt.ylim(0, 8)
    plt.tight_layout()
    plt.savefig(path_output+'sboxcompare_epoca_%.10d'%indice+'_erro_%09.4f'%erro+'.png')








def lerEntradas():
    obj0 = dirroot+'/maxmin_'+name_exp
    arq0 = open(obj0, 'r')
    texto0 = arq0.readlines()
    max=texto0[0].split(' ')
    min=texto0[1].split(' ')
    arq0.close()
    for i in range(0,len(min)):
        try:
            max[i] = float(max[i])
            min[i] = float(min[i])
        except:
            print "faio",max[i],min[i]



    obj1 = path_input+'_cabecalho'
    arq1 = open(obj1, 'r')
    texto1 = arq1.readlines()
    cabecalho=texto1[0].split(' ')   
    print cabecalho
    obj = path_input;
    arq = open(obj, 'r')
    texto = arq.readlines()
    random.shuffle(texto)#desordena entradas

    es=[]
    
  
    for l in range(0,len(texto)-1,1):

        e_s_i = []
        entrada=[]
        saida = []
        info=[]


        temp = texto[l].split(' ');
        
        info.append(temp[0])#data
        info.append(int(temp[1]))#mes
        info.append(float(temp[2]))#lat
        info.append(float(temp[3]))#lon    

        
        for i in range(0,len(temp)-1):
            e=0
            if (max[i] != min[i]):
                for v in variaveis_entrada.split(' '):
                    if cabecalho[i].find(v)>-1:
                        print ("encontrou",v)
                        e=(float(temp[i])-min[i])/(max[i] - min[i])
                        
                        entrada.append(e);


        #print entrada
        RedeNeural.max_output=max[-1]
        RedeNeural.min_output=min[-1]

        #print max[i],min[i]
        #print max[i]-min[i]
        e=(float(temp[-1])-min[-1])/(max[-1] - min[-1])
        saida.append(e);
        e_s_i.append(info)
        e_s_i.append(entrada)
        e_s_i.append(saida)
        es.append(e_s_i)
        #print es[l]
    arq.close()
    arq0.close()
    arq1.close()
    #print "\n\n\n"

    return es

def amostras():
    vetor=lerEntradas()
    treinamento=[]
    testes=[]
    validacao=[]
    ntrain=int(len(vetor)*1)
    ntest=int(len(vetor)*1)
    nvalid=int(len(vetor)*1)
    #print len(vetor)
    #print ntrain
    #print ntest
    #print nvalid

    i=0
    es=[]
    while i<ntrain:
        treinamento.append(vetor[i])
        testes.append(vetor[i])
        validacao.append(vetor[i])
        i+=1
    '''    
    while i<ntest+ntrain:
        testes.append(vetor[i])
        i+=1

    while i<ntest+ntrain+nvalid:
        validacao.append(vetor[i])
        i+=1
    '''
    es.append(treinamento)
    es.append(testes)
    es.append(validacao)
    #print len(treinamento),len(testes),len(validacao)

    return es

def iniciar():
    # Ensinar a rede a reconhecer o padrao XOR
    entradas_saidas = [
        [[0, 0], [0]],
        [[0, 1], [1]],
        [[1, 0], [1]],
        [[1, 1], [0]]
    ]

    es=amostras()
    entradas_saidas0=es[0]
    entradas_saidas1=es[1]
    entradas_saidas2=es[2]

    # cria rede neural com duas entradas, duas ocultas e um no de saida
    n = RedeNeural(len(entradas_saidas0[1][1]),N_NEURONIOS, len(entradas_saidas0[1][2]))
    #n = RedeNeural(2,4, 1)
    criar_linha()
    # treinar com os padrões
    n.treinar(entradas_saidas0,entradas_saidas1,entradas_saidas2)
    # testar
    criar_linha()
    n.test(entradas_saidas2)
    print "2 ocultos"

    #my_pickled_mary = pickle.dumps(n)
    #print (my_pickled_mary)
    

    #binary_file = open('rna3.bin',mode='wb')
    #my_pickled_mary = pickle.dump(n, binary_file)
    #binary_file.close()

if __name__ == '__main__':
    lerEntradas()
    iniciar()
    '''
    es=amostras()
    entradas_saidas0=es[0]
    entradas_saidas1=es[1]
    entradas_saidas2=es[2]
    print len(entradas_saidas0),len(entradas_saidas1),len(entradas_saidas2)
    
    file = open(path_output+'rna.bin','r')
    object_file = pickle.load(file)

   
    print "treinamento"
    object_file.treinar(entradas_saidas0,entradas_saidas1,entradas_saidas2)
    print "treinamento"
    criar_linha()
    #print "validacao"
    #object_file.valida(entradas_saidas2)

    #print "validacao", entradas_saidas2[30][0]    
    #print "validacao saida esperada",convertSaida(entradas_saidas2[30][1][0])
    #e = object_file.test2(entradas_saidas2[30][0])
    #print "validacao saida encontrada",convertSaida(e)
'''