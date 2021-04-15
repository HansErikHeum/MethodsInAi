from collections import defaultdict

import numpy as np

# Hans Erik Heum
# Problem 4 a,b, og c er i denne koden


class Variable:
    def __init__(self, name, no_states, table, parents=[], no_parent_states=[]):

        self.name = name
        self.no_states = no_states
        self.table = np.array(table)
        self.parents = parents
        self.no_parent_states = no_parent_states
        if self.table.shape[0] != self.no_states:
            raise ValueError(f"Number of states and number of rows in table must be equal. "
                             f"Recieved {self.no_states} number of states, but table has "
                             f"{self.table.shape[0]} number of rows.")

        if self.table.shape[1] != np.prod(no_parent_states):
            raise ValueError(
                "Number of table columns does not match number of parent states combinations.")

        if not np.allclose(self.table.sum(axis=0), 1):
            raise ValueError("All columns in table must sum to 1.")

        if len(parents) != len(no_parent_states):
            raise ValueError(
                "Number of parents must match number of length of list no_parent_states.")

    def __str__(self):
        """
        Pretty string for the table distribution
        For printing to display properly, don't use variable names with more than 7 characters
        """
        width = int(np.prod(self.no_parent_states))
        grid = np.meshgrid(*[range(i) for i in self.no_parent_states])
        s = ""
        for (i, e) in enumerate(self.parents):
            s += '+----------+' + '----------+' * width + '\n'
            gi = grid[i].reshape(-1)
            s += f'|{e:^10}|' + \
                '|'.join([f'{e + "("+str(j)+")":^10}' for j in gi])
            s += '|\n'

        for i in range(self.no_states):
            s += '+----------+' + '----------+' * width + '\n'
            state_name = self.name + f'({i})'
            s += f'|{state_name:^10}|' + \
                '|'.join([f'{p:^10.4f}' for p in self.table[i]])
            s += '|\n'

        s += '+----------+' + '----------+' * width + '\n'

        return s

    def probability(self, state, parentstates):

        if not isinstance(state, int):
            raise TypeError(
                f"Expected state to be of type int; got type {type(state)}.")
        if not isinstance(parentstates, dict):
            raise TypeError(
                f"Expected parentstates to be of type dict; got type {type(parentstates)}.")
        if state >= self.no_states:
            raise ValueError(
                f"Recieved state={state}; this variable's last state is {self.no_states - 1}.")
        if state < 0:
            raise ValueError(
                f"Recieved state={state}; state cannot be negative.")

        table_index = 0
        for variable in self.parents:
            if variable not in parentstates:
                raise ValueError(
                    f"Variable {variable.name} does not have a defined value in parentstates.")

            var_index = self.parents.index(variable)
            table_index += parentstates[variable] * \
                np.prod(self.no_parent_states[:var_index])
        return self.table[state, int(table_index)]


class BayesianNetwork:

    def __init__(self):
        # All nodes start out with 0 edges
        self.edges = defaultdict(lambda: [])
        self.variables = {}                   # Dictionary of "name":TabularDistribution

    def add_variable(self, variable):
        """
        Adds a variable to the network.
        """
        if not isinstance(variable, Variable):
            raise TypeError(f"Expected {Variable}; got {type(variable)}.")
        self.variables[variable.name] = variable

    def add_edge(self, from_variable, to_variable):
        """
        Adds an edge from one variable to another in the network. Both variables must have
        been added to the network before calling this method.
        """
        if from_variable not in self.variables.values():
            raise ValueError(
                "Parent variable is not added to list of variables.")
        if to_variable not in self.variables.values():
            raise ValueError(
                "Child variable is not added to list of variables.")
        self.edges[from_variable].append(to_variable)

    def sorted_nodes(self):
        """
        TODO: Implement Kahn's algorithm (or some equivalent algorithm) for putting
              variables in lexicographical topological order.
        Returns: List of sorted variable names.
        """
    # Her lager jeg bare en ny dictionary med variabelen som nøkkel
    # Denne inneholder et tall som sier hvor mange kanter som går til noden
        dictionary_edges_count = {}
        for key in self.variables:
            dictionary_edges_count[self.variables[key]] = 0
            # Her lager jeg bare en dictionary med hvor mange kanter som går til en node
        for key in self.edges:
            for element in self.edges[key]:
                existing_value = dictionary_edges_count[element]
                dictionary_edges_count[element] = existing_value+1

        # køen er en liste med noder som ikke har noen kanter inn til seg
        # En node uten kanter blir dermed lagt inn i køen
        queue = []
        for key in dictionary_edges_count:
            if dictionary_edges_count[key] == 0:
                queue.append(key)
        topological_sorted = []
        # algoritmen kjører helt til køen er tom
        while ((len(queue) != 0)):
            at = queue.pop()
            topological_sorted.append(at)
            for key in self.edges:
                if key == at:
                    for element in self.edges[key]:
                        dictionary_edges_count[element] -= 1
                        if dictionary_edges_count[element] == 0:
                            queue.append(element)
        # Fordi self.bayesian_network.variables er en dictionary, som jeg har brukt i senere kode
        # velger jeg å returnere en topologisk sortert dictionary, istedenfor en liste
        returnable_dict = {}
        for i in topological_sorted:
            returnable_dict[i.name] = i
        return(returnable_dict)
        # Jeg har valgt å ikke sjekke om grafen inneholder sykler, da det sto på Piazza at dette ikke var nødvendig
        # Det kunne lett blitt implementert ved å sjekke at lengden på den topologiske sorterte listen samsvarer med antall noder
        # I det bayesiske nettverket.


class InferenceByEnumeration:
    def __init__(self, bayesian_network):
        self.bayesian_network = bayesian_network
        self.topo_order = bayesian_network.sorted_nodes()

    def _enumeration_ask(self, X, evidence):
        # TODO: Implement Enumeration-Ask algortihm as described in Problem 4 b)

        # Q_x er en liste der vi etterhvert legger inn alle sannsynlighetene for variablens mulige states
        Q_x = []
        states = self.bayesian_network.variables[X].no_states
        # for hver verdi X kan ha regner vi ut sannsynligheten for at det skjer
        for i in range(states):
            # Kopierer bevisene og legger til Variabelen med den gitte staten

            new_evidence = evidence.copy()
            new_evidence[X] = i
            # sender inn variablene og det nye beviset inn i enumerate_all
            probability_to_append = self._enumerate_all(
                self.topo_order, new_evidence)
            # Etter utregningen legger vi inn sannsynligheten i en tabell
            Q_x.append(probability_to_append)
        # Må normalisere sannsynligheten, ved å bruke formelen:
        # alpha=1/(summen av sannsynlighetene)
        alpha_sum = 0
        # summerer alle sannsynlighetene
        for i in range(len(Q_x)):
            alpha_sum += Q_x[i]
        alpha = (1/alpha_sum)
        # må gjøre det om til et numpy array for at reshape-funksjonen skal fungere
        numpy_Q_x = np.array(Q_x)
        # ganger inn alphaen
        finished_normalized = alpha*numpy_Q_x

        return finished_normalized
    # Det meste av det jeg gjør i denne algoritmen er ganske tydelig utifra oppgitt pseudokode
    # Der det kanskje er uklart har jeg prøvd å kommentere

    def _enumerate_all(self, vars, evidence):
        # TODO: Implement Enumerate-All algortihm as described in Problem 4 b)

        if len(vars) == 0:
            return 1
        Y = next(iter(vars))
        variables_removed = vars.copy()
        del variables_removed[Y]

        if Y in evidence.keys():
            # må lage en dictionary som skal sendes i probability funkjsonen
            # foreldrene MÅ være i evidence dictionarien
            parents_dict = {}
            for parent in self.bayesian_network.variables[Y].parents:
                if parent in evidence.keys():
                    parents_dict[parent] = evidence[parent]
            return (self.bayesian_network.variables[Y].probability(evidence[Y], parents_dict) * self._enumerate_all(variables_removed, evidence))

        else:
            sum = 0
            parents_dict = {}
            for parent in self.bayesian_network.variables[Y].parents:
                if parent in evidence.keys():
                    parents_dict[parent] = evidence[parent]
            for i in range(self.bayesian_network.variables[Y].no_states):

                evidence2 = evidence.copy()
                evidence2[Y] = i

                sum += (self.bayesian_network.variables[Y].probability(
                    i, parents_dict) * self._enumerate_all(variables_removed, evidence2))
            return sum

    def query(self, var, evidence={}):
        """
        Wrapper around "_enumeration_ask" that returns a
        Tabular variable instead of a vector
        """

        q = self._enumeration_ask(var, evidence).reshape(-1, 1)

        return Variable(var, self.bayesian_network.variables[var].no_states, q)


def problem3c():
    # Disse eksmeplene er kun laget for å vise at koden fungerer
    d1 = Variable('A', 2, [[0.8], [0.2]])
    d2 = Variable('B', 2, [[0.5, 0.2],
                           [0.5, 0.8]],
                  parents=['A'],
                  no_parent_states=[2])
    d3 = Variable('C', 2, [[0.1, 0.3],
                           [0.9, 0.7]],
                  parents=['B'],
                  no_parent_states=[2])
    d4 = Variable('D', 2, [[0.6, 0.8],
                           [0.4, 0.2]],
                  parents=['B'],
                  no_parent_states=[2])

    print(f"Probability distribution, P({d1.name})")
    print(d1)

    print(f"Probability distribution, P({d2.name} | {d1.name})")
    print(d2)

    print(f"Probability distribution, P({d3.name} | {d2.name})")
    print(d3)

    print(f"Probability distribution, P({d4.name} | {d2.name})")
    print(d4)

    bn = BayesianNetwork()
    # La de inn med forskjellig rekkefølge for å sjekke at den topologiske sortering fungerer som den skal
    bn.add_variable(d4)
    bn.add_variable(d3)
    bn.add_variable(d2)
    bn.add_variable(d1)
    bn.add_edge(d1, d2)
    bn.add_edge(d2, d3)
    bn.add_edge(d2, d4)

    inference = InferenceByEnumeration(bn)
    posterior = inference.query('C', {'D': 1})
    posterior2 = inference.query('A', {'C': 1, 'D': 0})

    print(f"Probability distribution, P({d3.name} | !{d4.name})")
    print(posterior)
    print(f"Probability distribution, P({d1.name} | !{d3.name}  | {d4.name})")
    print(posterior2)


def monty_hall():
    # TODO: Implement the monty hall problem as described in Problem 4c)
    # P(Prize | ChosenByGuest = 1, OpenedByHost =3)
    # Variablene må være:
    # Døren prisen ligger i
    # Døren gjesten velger
    # Døren hosten åpner

    # Selv om det er dør (1,2,3), bruker jeg heller statene (0,1,2)
    # Det vil si at dør 0, er egt dør 1 , osv.
    # Det er 1/3 sjanse hvilken dør prisen ligger i
    d1 = Variable('Prize', 3, [[1/3], [1/3], [1/3]])
    print(f"Probability distribution, P({d1.name})")
    print(d1)
    # Det er også 1/3 sjanse for hver dør gjesten velger
    d2 = Variable('Chosen', 3, [[1/3], [1/3], [1/3]])
    print(f"Probability distribution, P({d2.name})")
    print(d2)
    # Denne tabellen er laget vha logikk, hvis prisen er i dør 2, og dør 1 er valgt, kommer hosten uansett til å åpne dør 3, osv

    d3 = Variable('Host', 3, [[0, 0, 0, 0, 0.5, 1, 0, 1, 0.5], [0.5, 0, 1, 0, 0, 0, 1, 0, 0.5], [
                  0.5, 1, 0, 1, 0.5, 0, 0, 0, 0]], ['Prize', 'Chosen'], [3, 3])

    print(f"Probability distribution, P({d3.name} | {d1.name} | {d2.name})")
    print(d3)

    # Lager et Bayesian nettverk, på samme måte som i forrige oppgave
    bn2 = BayesianNetwork()

    bn2.add_variable(d1)
    bn2.add_variable(d2)
    bn2.add_variable(d3)
    bn2.add_edge(d1, d3)
    bn2.add_edge(d2, d3)

    inference2 = InferenceByEnumeration(bn2)
    posterior2 = inference2.query('Prize', {'Chosen': 0, 'Host': 2})

    print(
        f"Probability distribution, P({d1.name} | {d2.name}=1  , {d3.name}=3)")
    print(posterior2)
    print("Som en kan se på tabellen, vil det lønne seg å bytte dør")


if __name__ == '__main__':
    problem3c()
    monty_hall()
