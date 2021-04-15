import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


class Matrix:
    def __init__(self, transition_matrix, initial_matrix, evidence_matrix_true, evidence_matrix_false, evidence):
        self.transition = transition_matrix
        self.initial = initial_matrix
        self.evidence_true = evidence_matrix_true
        self.evidence_false = evidence_matrix_false
        self.evidence = evidence

    def __str__(self):
        pass


# med np.array kan man benytte seg av dot() og transform()

# This is a filtering proble, which also includes prediction
# I shall now return a list with the different probilities,
# with the different T's, aka
# P(x_1 | e_1), P(x_2 | e_2,e_1),
# P(x_3 | e_3,e_2)  <-- den blir vel kun påvirket av tidligere evidence
# Blir vel da etterhvert slik:
# [<0.8,0,2> , <0.3,0.7>] (Et element for hver t!)
# f_1_t+1 = alpha * O_t+1 * T(transformert)*f_1:t
class Algorithm:
    def __init__(self, matrix):
        self.matrix = matrix

    def pre_forward(self, startValue, length, hasEvidence):
        probability = [self.matrix.initial]

        if hasEvidence:
            index = 0
            while index != (length):
                probability.append(
                    self.forward(probability[index], self.matrix.evidence[index]))
                index += 1

        # If evidence dict is empty
        # This is for c
        else:
            index = 0
            counter = startValue
            while counter != (length):
                probability.append(
                    self.forward(probability[index], 0))
                index += 1
                counter += 1

        return probability

    def forward(self, f1_t, evidence_value):
        forward_dict = None
        if evidence_value == 0:  # is the evidence true or false
            forward_dict = (np.dot(np.dot(self.matrix.evidence_true,
                                          np.transpose(self.matrix.transition)), f1_t))
        else:
            forward_dict = np.dot(np.dot(self.matrix.evidence_false,
                                         np.transpose(self.matrix.transition)), f1_t)
        # Må normalisere den
         # Må normalisere sannsynligheten, ved å bruke formelen:
        # alpha=1/(summen av sannsynlighetene)
        alpha_sum = 0
        # summerer alle sannsynlighetene
        for i in range(len(forward_dict)):
            alpha_sum += forward_dict[i]
        alpha = (1/alpha_sum)
        # må gjøre det om til et numpy array for at reshape-funksjonen skal fungere
        # ganger inn alphaen
        finished_normalized = alpha*forward_dict
        return finished_normalized

    def pre_backward(self, startValue, length):
        # probability = np.zeros((length, 2, 2))
        probability = np.zeros((length+1, 2))
        # backwardoppgaven skal kun hvære med fra e_5, bruker derfor e_6 som initial value

        probability[length] = [1, 1]
        # iterer bakerst også nedover
        for i in range(length-1, -1, -1):  # 4,3,2,1,0
            probability[i] = self.backward(
                probability[i+1], self.matrix.evidence[i])
        return probability

    def backward(self, b_t, evidence_value):
        if evidence_value == 0:
            value = np.dot(np.dot(self.matrix.transition,
                                  self.matrix.evidence_true), b_t)
        else:
            value = np.dot(np.dot(self.matrix.transition,
                                  self.matrix.evidence_false), b_t)
        alpha_sum = 0
        for i in range(len(value)):
            alpha_sum += value[i]
        alpha = (1/alpha_sum)
        finished_normalized = alpha*value
        return finished_normalized
        # return value

    def forward_backward(self, f_values, b_values):
        probabilities = []

        print("Her ser du hva som blir ganget sammen i forward_backward algoritmen: \n")
        for i in range(len(b_values)):
            # [0.5 , 0.5] * [0.43434 , 0.43434]
            value = f_values[i]*b_values[i]
            print("f_values * b_values")
            print(f_values[i], " * ", b_values[i])
            # The wikipage states the b_value should be normalized as long you use the normalized
            # value in the next iteration.
            # On Piazza, however, it's said that we should not normalize each backward algorithm step.
            # I chose to follow wikipedias algorithm, as I could easy validate my algorithm through
            # testing with their values.

            alpha_sum = 0
            for i in range(len(value)):
                alpha_sum += value[i]
            alpha = (1/alpha_sum)
            finished_normalized = alpha*value
            probabilities.append(finished_normalized)
        print("slik blir da fasiten på forward backward:  \n ", probabilities)
        return probabilities

    def normalize(self, array):
        alpha_sum = 0
        for i in range(len(array)):
            alpha_sum += array[i]
        alpha = (1/alpha_sum)
        finished_normalized = alpha*array
        return finished_normalized

    def Viterbi(self, evidence_probabilities_true, evidence_probabilities_false, task):
        evidence = self.matrix.evidence  # called 'o' in the pseudo
        states = [0, 1]  # called s in the pseudo 0 is true, 1 is false
        Tm = self.matrix.transition
        # matrix 6 x 2
        trellis = np.zeros((2, len(evidence)))
        backpointers = np.zeros((2, len(evidence)))
        # forstår ikke helt om jeg må normalisere disse eller ikke.
        first_probabilities = np.array([self.matrix.initial[0] *
                                        self.matrix.evidence_true[0, 0], self.matrix.initial[1] *
                                        self.matrix.evidence_true[1, 1]])

        normalized = self.normalize(first_probabilities)
        trellis[0, 0] = normalized[0]
        trellis[1, 0] = normalized[1]
        # for hver observasjonß
        for o in range(1, len(evidence)):
            # for hver state
            for s in range(len(states)):
                # nå må jeg finne argmax, som er indeksen på den k'en som gir størst verdi?
                # hvis evidence er true
                if evidence[o] == 0:

                    k_true = trellis[0, o-1]*Tm[0, s] * \
                        evidence_probabilities_true[s]
                    k_false = trellis[1, o-1]*Tm[1, s] * \
                        evidence_probabilities_true[s]
                if evidence[o] == 1:
                    k_true = trellis[0, o-1]*Tm[0, s] * \
                        evidence_probabilities_false[s]
                    k_false = trellis[1, o-1]*Tm[1, s] * \
                        evidence_probabilities_false[s]

                if k_true > k_false:
                    best_k = 0
                    k_value = k_true
                else:
                    best_k = 1
                    k_value = k_false
                if evidence[o] == 0:

                    trellis[s, o] = trellis[best_k, o-1] * \
                        Tm[best_k, s]*evidence_probabilities_true[s]
                else:
                    trellis[s, o] = trellis[best_k, o-1] * \
                        Tm[best_k, s]*evidence_probabilities_false[s]
        best_path = []
        for i in range(len(trellis[0])):
            if trellis[0, i] > trellis[1, i]:
                best_path.append("True")
            else:
                best_path.append("False")
        print("Here comes the solution for task 1e):  \n")
        print("\n")
        print("The trellis looks like this:  \n", trellis)
        print("best path: \n", best_path)
        print("____________________________________")


def plottingFunction(probabilities, fromT, toT, title):
    true_probabilities = []
    # for i in probabilities:
    # true_probabilities.append(i[0])

    for i in range(toT-fromT):
        true_probabilities.append(probabilities[i][0])
    t_values = np.arange(fromT, toT)
    plt.plot(range(fromT, toT), true_probabilities)
    plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(1))
    plt.xlabel('X_t')
    plt.ylabel('Probability of X=True')
    plt.title(title)
    plt.show()


def task1b():
    transition_matrix = np.array([[0.8, 0.2], [0.3, 0.7]])
    start_matrix = np.array([0.5, 0.5])
    evidence_matrix_true = np.array([[0.75, 0], [0, 0.2]])
    evidence_matrix_false = np.array([[0.25, 0], [0, 0.8]])
    # 0=true, 1=false
    evidence = [0, 0, 1, 0, 1, 0]
    matrices = Matrix(transition_matrix, start_matrix,
                      evidence_matrix_true, evidence_matrix_false, evidence)
    algorithm = Algorithm(matrices)
    distribution = algorithm.pre_forward(0, 6, True)
    return distribution


def task1c():
    transition_matrix = np.array([[0.8, 0.2], [0.3, 0.7]])
    previous_probabilities = task1b()
    # bruker den siste sannsynligheten fra forrige oppgave som initial_matrix
    start_matrix = previous_probabilities[len(previous_probabilities)-1]
    # siden vi ikke har noen bevis setter vi bare sannsynlighetene til 1,1.
    evidence_matrix_true = np.array([[1, 0], [0, 1]])
    evidence_matrix_false = np.array([[1, 0], [0, 1]])
    # 0=true, 1=false
    evidence = None
    matrices = Matrix(transition_matrix, start_matrix,
                      evidence_matrix_true, evidence_matrix_false, evidence)
    algorithm = Algorithm(matrices)

    distribution = algorithm.pre_forward(7, 30, False)
    plottingFunction(distribution, 7, 30, "Task 1c")
    return distribution


def task1d():
    # bruker formelen for smoothing i denne oppgaven, altså
    # alfa * f_1:k * b_k+1:t
    # Listen for f_1:k har vi allerede fra oppgave 1b, trenger derfor kun å lage en slik liste for backwarding også
    transition_matrix = np.array([[0.8, 0.2], [0.3, 0.7]])
    start_matrix = np.array([0.5, 0.5])
    f_values = task1b()
    evidence_matrix_true = np.array([[0.75, 0], [0, 0.2]])
    evidence_matrix_false = np.array([[0.25, 0], [0, 0.8]])
    evidence = [0, 0, 1, 0, 1, 0]
    matrices = Matrix(transition_matrix, start_matrix,
                      evidence_matrix_true, evidence_matrix_false, evidence)
    algorithm = Algorithm(matrices)
    b_values = algorithm.pre_backward(0, 6)
    smoothed_values = algorithm.forward_backward(f_values, b_values)
    plottingFunction(smoothed_values, 0, 6, "Task 1d")


def task1e():
    transition_matrix = np.array([[0.8, 0.2], [0.3, 0.7]])
    start_matrix = np.array([0.5, 0.5])
    evidence_matrix_true = np.array([[0.75, 0], [0, 0.2]])
    evidence_matrix_false = np.array([[0.25, 0], [0, 0.8]])
    # 0=true, 1=false
    evidence = [0, 0, 1, 0, 1, 0]
    matrices = Matrix(transition_matrix, start_matrix,
                      evidence_matrix_true, evidence_matrix_false, evidence)
    algorithm = Algorithm(matrices)
    evidence_probabilities_true = [0.75, 0.2]
    evidence_probabilities_false = [0.25, 0.8]
    algorithm.Viterbi(evidence_probabilities_true,
                      evidence_probabilities_false, "oppgave 1 e")


def wikipage():
    # implementerer algoritmen fra wikipediasiden, for å dobbeltsjekke at funksjonen stemmer
    # Ser i etterkant at tallene stemmer med wikipedia, og algoritmen er derfor korrekt
    transition_matrix = np.array([[0.7, 0.3], [0.3, 0.7]])
    start_matrix = np.array([0.5, 0.5])
    f_values = umbrella_task()
    evidence_matrix_true = np.array([[0.9, 0], [0, 0.2]])
    evidence_matrix_false = np.array([[0.1, 0], [0, 0.8]])
    evidence = [0, 0, 1, 0, 0]
    matrices = Matrix(transition_matrix, start_matrix,
                      evidence_matrix_true, evidence_matrix_false, evidence)
    algorithm = Algorithm(matrices)
    b_values = algorithm.pre_backward(0, 5)
    smoothed_values = algorithm.forward_backward(f_values, b_values)


def viterbiWiki():
    transition_matrix = np.array([[0.7, 0.3], [0.3, 0.7]])
    start_matrix = np.array([0.5, 0.5])
    evidence_matrix_true = np.array([[0.9, 0], [0, 0.2]])
    evidence_matrix_false = np.array([[0.1, 0], [0, 0.8]])
    evidence = [0, 0, 1, 0, 0]
    matrices = Matrix(transition_matrix, start_matrix,
                      evidence_matrix_true, evidence_matrix_false, evidence)
    algorithm = Algorithm(matrices)
    evidence_probabilities_true = [0.9, 0.2]
    evidence_probabilities_false = [0.1, 0.8]
    algorithm.Viterbi(evidence_probabilities_true,
                      evidence_probabilities_false, "wikiPage")


def umbrella_task():
    transition_matrix = np.array([[0.7, 0.3], [0.3, 0.7]])
    start_matrix = np.array([0.5, 0.5])
    evidence_matrix_true = np.array([[0.9, 0], [0, 0.2]])
    evidence_matrix_false = np.array([[0.1, 0], [0, 0.8]])
    # 0=true, 1=false
    evidence = [0, 0, 1, 0, 1, 0]
    matrices = Matrix(transition_matrix, start_matrix,
                      evidence_matrix_true, evidence_matrix_false, evidence)
    algorithm = Algorithm(matrices)
    return algorithm.pre_forward(0, 6, True)


# umbrella_task()  # skulle bare dobbeltsjekke at denne stemte!
distribution = task1b()
plottingFunction(distribution, 0, 7, "Task 1b")
task1c()
task1d()
# viterbiWiki() #måtte dobbeltsjekke algoritmen med gitte tallverdier
task1e()
