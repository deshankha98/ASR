import pygtrie
import torch
# from multiprocessing import Pool, Lock, Process
import heapq
import numpy as np
from models.ModelFactory import ModelFactory
import math
from metaclasses.Singleton import Singleton
import copy


STARTING_STATE = '*'


class BeamSearchContext():
    def __init__(self, prefix_char, log_prob, epsilon_added, state, sentence_formed='', state_trace=[], new_word_start=False):
        self.prefix = prefix_char
        self.log_prob = log_prob
        self.state = state
        self.epsilon_added = epsilon_added
        self.sentence_formed = sentence_formed
        self.state_trace = state_trace
        self.new_word_start = new_word_start
    def create_key(self):
        sentence_formed = self.sentence_formed + ":" + self.prefix
        epsilon_added = ":TRUE" if self.epsilon_added else ":FALSE"
        state = ":" + self.state
        return sentence_formed + epsilon_added + state

    @staticmethod
    def decode_key_back_to_context(beam_context_key, log_prob, state_trace):
        sentence_formed = beam_context_key.split(':')[0]
        prefix = beam_context_key.split(':')[1]
        epsilon_added_str = beam_context_key.split(':')[2]
        epsilon_added = True if epsilon_added_str == "TRUE" else False
        state = beam_context_key.split(':')[3]
        return BeamSearchContext(prefix, log_prob, epsilon_added, state, sentence_formed=sentence_formed, state_trace=state_trace)


def extract_model():
    return ModelFactory().get_model()


class CustomCorpusManager(metaclass=Singleton):
    def __init__(self, states, state_transition, state_vs_words):
        self.state_wise_trie = {}
        for state in states:
            self.state_wise_trie[state] = make_trie(state_vs_words[state])
        self.state_transition = state_transition

    def get_children_chars(self, prefix: str, curr_state: str, epsilon_added):

        possible_next_states = self.state_transition[curr_state] if curr_state in self.state_transition else []
        try:
            list_of_child_chars = []
            if len(prefix) == 0:
                list_of_child_chars.append((epsilon, curr_state, False))
                list_of_child_chars.append((space, curr_state, False))
                for state in possible_next_states:
                    corpus: pygtrie.CharTrie = self.state_wise_trie[state]
                    node, _ = corpus._get_node(epsilon)
                    children = node.children
                    if len(children) == 1:  # OneChild Object
                        list_of_child_chars.append((children.step, state, True))
                    else:
                        for child_char in children:
                            list_of_child_chars.append((child_char, state, True))
                return list_of_child_chars

            char_trie = self.state_wise_trie[curr_state]
            node, _ = char_trie._get_node(prefix)
            children = node.children

            if char_trie.has_key(prefix):
                list_of_child_chars.append((space, curr_state, False))
                list_of_child_chars.append((epsilon, curr_state, False))
                for state in possible_next_states:
                    next_state_corpus: pygtrie.CharTrie = self.state_wise_trie[state]
                    next_state_node, _ = next_state_corpus._get_node(epsilon)
                    next_state_children = next_state_node.children
                    if len(next_state_children) == 1:  # OneChild Object
                        list_of_child_chars.append((next_state_children.step, state, True))
                    else:
                        for child_char in next_state_children:
                            list_of_child_chars.append((child_char, state, True))
                if len(children) != 0:
                    if len(children) == 1:  # OneChild Object
                        list_of_child_chars.append((children.step, curr_state, False))
                    else:
                        for child_char in children:
                            list_of_child_chars.append((child_char, curr_state, False))
                if not epsilon_added:
                    last_char = prefix[-1]
                    list_of_child_chars.append((last_char, curr_state, False))
                return list_of_child_chars

            # if self.has_key(prefix):  # key word exists
            #     list_of_child_chars.append(space)
            #     list_of_child_chars.append(epsilon)
            #     if len(children) == 1:  # OneChild Object
            #         list_of_child_chars.append(children.step)
            #     else:
            #         for child_char in children:
            #             list_of_child_chars.append(child_char)
            #     return list_of_child_chars

            # else a normal prefix
            list_of_child_chars.append((epsilon, curr_state, False))
            if not epsilon_added:
                last_char = prefix[-1]
                list_of_child_chars.append((last_char, curr_state, False))
            if len(children) == 1:  # OneChild Object
                list_of_child_chars.append((children.step, curr_state, False))
            else:
                for child_char in children:
                    list_of_child_chars.append((child_char, curr_state, False))
            return list_of_child_chars

        except KeyError:
            raise KeyError("Node not found for prefix %s", prefix)

    def check_terminal_word(self, beam_element: BeamSearchContext):
        if 'terminal_states' in self.state_transition and beam_element.state in self.state_transition['terminal_states']:
            if self.state_wise_trie[beam_element.state].has_key(beam_element.prefix):
                return True
        return False
    def check_for_acceptance(self, beam_element: BeamSearchContext):
        if self.check_terminal_word(beam_element):
            return True
        if 'accepted_states' in self.state_transition and beam_element.state in self.state_transition['accepted_states']:
            if self.state_wise_trie[beam_element.state].has_key(beam_element.prefix):
                return True
        return False



def make_trie(words: list[str]):
    corpus = pygtrie.CharTrie()
    for word in words:
        corpus[word] = True
    return corpus


# reformat st prob(epsilon) = prob(blank_char_index) + prob(quote_char_index)
epsilon = ''
space = '|'

def get_emission_prob(token, emissions):
    _, processor = extract_model()
    if token == space:
        return emissions[processor.tokenizer.convert_tokens_to_ids(space)]

    if token == epsilon:
        return emissions[processor.tokenizer.pad_token_id]
    ### convert token to lower case
    return emissions[processor.tokenizer.convert_tokens_to_ids(token.lower())]


def join_token(beam_element: BeamSearchContext, child_beam_element: BeamSearchContext):
    updated_log_prob = beam_element.log_prob + child_beam_element.log_prob
    prefix = beam_element.prefix
    child_char = child_beam_element.prefix
    state = beam_element.state
    next_state = child_beam_element.state
    epsilon_added = beam_element.epsilon_added
    sentence_formed = copy.deepcopy(beam_element.sentence_formed)
    state_trace = beam_element.state_trace
    new_word_start = child_beam_element.new_word_start
    updated_state_trace = copy.deepcopy(state_trace)

    if child_char == epsilon:
        return BeamSearchContext(prefix, updated_log_prob, True, state, sentence_formed=sentence_formed, state_trace=updated_state_trace)
    if child_char == space:
        if len(prefix) == 0:
            return BeamSearchContext('', updated_log_prob, True, state, sentence_formed=sentence_formed, state_trace=updated_state_trace) ### it should never go into this condition
        if len(prefix) > 0:
            return BeamSearchContext(prefix, updated_log_prob, True, state, sentence_formed=sentence_formed, state_trace=updated_state_trace)


    if len(prefix) == 0:
        updated_state_trace.append(next_state)
        return BeamSearchContext(child_char, updated_log_prob, False, next_state, sentence_formed=sentence_formed, state_trace=updated_state_trace)

    if len(prefix) > 0 and new_word_start:
        updated_state_trace.append(next_state)
        return BeamSearchContext(child_char, updated_log_prob, False, next_state, sentence_formed=sentence_formed + prefix, state_trace=updated_state_trace)
    if len(prefix) > 0 and (not new_word_start):
        if epsilon_added:
            updated_state_trace.append(next_state)
            updated_prefix = prefix + child_char
        else:
            if prefix[-1] != child_char:
                updated_state_trace.append(next_state)
            updated_prefix = prefix if prefix[-1] == child_char else prefix + child_char
        return BeamSearchContext(updated_prefix, updated_log_prob, False, state, sentence_formed=sentence_formed, state_trace=updated_state_trace)


BEAM_SIZE = 100
top_k = 3

def beam_search_update(beam_element: BeamSearchContext, emissions_current, common_temp_beam, corpus_manager):
    prefix = beam_element.prefix
    state = beam_element.state
    epsilon_added = beam_element.epsilon_added
    children = corpus_manager.get_children_chars(prefix, state, epsilon_added)
    allowed_tokens = []
    for child, child_state, new_word_start in children:
        allowed_tokens.append(BeamSearchContext(child, get_emission_prob(child, emissions_current), None, child_state, new_word_start=new_word_start))
    # allowed_tokens = normalize(allowed_tokens)
    for i in range(len(allowed_tokens)):
        updated_beam_element = join_token(beam_element, allowed_tokens[i])
        # if has_key(common_temp_beam, updated_prefix) is False or common_temp_beam[
        #     updated_prefix] < updated_prob:
        #     common_temp_beam[updated_prefix] = updated_prob

        if has_key(common_temp_beam, updated_beam_element) is False:
            common_temp_beam[updated_beam_element.create_key()] = (updated_beam_element.log_prob, updated_beam_element.state_trace)
        else:
            key = updated_beam_element.create_key()
            common_temp_beam[key] = (log_added_probs(common_temp_beam[key][0], updated_beam_element.log_prob), updated_beam_element.state_trace)

def log_added_probs(log_prob1, log_prob2):
    ### return log(p1 + p2) given logp1 logp2
    diff1 = log_prob1 - log_prob2
    diff2 = log_prob2 - log_prob1
    if diff1 > 0:
        return log_prob2 + math.log(1 + math.exp(diff1))
    if diff2 > 0:
        return log_prob1 + math.log(1 + math.exp(diff2))
    return math.log(2) + log_prob1

def has_key(beam, beam_element: BeamSearchContext):
    try:
        key = beam_element.create_key()
        return beam[key] is not None
    except KeyError:
        return False


class GuidedBeamSearch():
    def __init__(self, emissions, corpus: CustomCorpusManager, states: list, state_transition, state_vs_words):
        self.emissions = emissions
        self.corpus_manager = corpus
        self.common_temp_beam = {}
        self.reverse_common_temp_beam = {}
        # self.p = make_pool()
        self.detected_sentence = None
        self.reverse_detected_words = []
        self.detected_prefixes = []
        self.reverse_detected_prefixes = []
        self.emissions_current = emissions[0]
        self.reverse_emissions_current = emissions[-1]
        if self.corpus_manager == None:
            self.corpus_manager = CustomCorpusManager(states, state_transition, state_vs_words)
        # self.lock = Lock()

    def guided_beam_search(self, logging=False, file_name=''):
        emissions = self.emissions
        last_detected_prefixes = [BeamSearchContext('', 0.0, True, STARTING_STATE)]
        temp_detected_prefixes = []
        for i in range(emissions.shape[0]):
            if self.detected_sentence is not None:
                break
            self.emissions_current = emissions[i]
            self.common_temp_beam.clear()

            for j in range(len(last_detected_prefixes)):
                beam_search_update(last_detected_prefixes[j], self.emissions_current, self.common_temp_beam, self.corpus_manager)

            last_detected_prefixes.clear()
            temp_detected_prefixes.clear()

            ctr = 0
            for beam_element_key, (log_prob, state_trace) in sorted(self.common_temp_beam.items(), key=lambda x: x[1][0], reverse=True):
                if ctr == BEAM_SIZE:
                    break

                beam_element = BeamSearchContext.decode_key_back_to_context(beam_element_key, log_prob, state_trace)
                last_detected_prefixes.append(beam_element)
                temp_detected_prefixes.append((beam_element.sentence_formed, beam_element.prefix, beam_element.log_prob))
                ctr += 1
            self.detected_prefixes.append(temp_detected_prefixes)
        #### check for possible word formation
        if self.detected_sentence is not None:
            return self.detected_prefixes, self.detected_sentence
        for beam_element in last_detected_prefixes:
            if self.corpus_manager.check_for_acceptance(beam_element):
                self.detected_sentence = beam_element.sentence_formed + beam_element.prefix
                break
        return self.detected_prefixes, self.detected_sentence
