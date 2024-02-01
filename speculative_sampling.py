import torch
from utils import sample_from_draft_model, get_distribution, sample
from transformers import AutoTokenizer

# def speculative_sampling(target_model, draft_model, initial_prompt_seq, max_new_tokens, tokenizer, lookahead=4, temperature=1.0, debug=True):
#     '''
#     Implementation of Algorithm 2 of the paper - Accelerating Large Language Model Decoding 
#     with Speculative Sampling (https://arxiv.org/abs/2302.01318)
#     '''
#     assert initial_prompt_seq.shape[0] == 1, 'Batch size should be 1'

#     n = initial_prompt_seq.shape[-1]
#     fin_prompt_seq = initial_prompt_seq.detach().clone()

#     while n < max_new_tokens:
#         n_orig = n
#         N = fin_prompt_seq.shape[-1]
#         draft_outputs, draft_logits = sample_from_draft_model(draft_model, fin_prompt_seq, new_tokens=lookahead, temperature=temperature)
#         draft_outputs2, draft_logits2 = sample_from_draft_model(draft_model, fin_prompt_seq, new_tokens=lookahead, temperature=temperature)

#         # print(draft_outputs, draft_outputs2)
        
#         if debug:
#             print(f"Possible continuations: {tokenizer.decode(draft_outputs[0,n_orig:], skip_special_tokens=True)}")

#         draft_outputs_list 

#         target_logits = target_model(draft_outputs).logits[:, -lookahead-1:, :]

#         target_model_distribution = get_distribution(target_logits, temperature)
#         draft_model_distribution = get_distribution(draft_logits, temperature)

#         accepted_flag = 1
        
#         for t in range(lookahead):
#             numerator = target_model_distribution[:, t, draft_outputs[0, N+t]]
#             denominator = draft_model_distribution[:, t, draft_outputs[0, N+t]]
#             ratio = (numerator / denominator)
#             uniform_distribution = torch.rand_like(numerator)
#             ones_tensor = torch.ones_like(numerator)

#             # Rejection Sampling
#             ## Acceptance
#             if (uniform_distribution < torch.min(ones_tensor, ratio)).any():
#                 fin_prompt_seq = torch.concat([fin_prompt_seq, draft_outputs[:, N+t].unsqueeze(dim=-1)], dim=-1)
#                 n += 1

#             ## Rejection
#             else:
#                 new_dist = (target_model_distribution[:, t, :] - draft_model_distribution[:, t, :])
#                 new_dist = torch.max(torch.zeros_like(new_dist), new_dist)
#                 new_dist = new_dist / new_dist.sum(dim=-1, keepdim=True)
#                 token_id = torch.multinomial(new_dist, num_samples=1)[0]
#                 fin_prompt_seq = torch.concat([fin_prompt_seq, token_id[None,...]], dim=-1)
#                 accepted_flag = 0
#                 break

#         if accepted_flag == 1:
#             sample_token = sample(target_logits[:, -1, :], temperature=temperature)
#             fin_prompt_seq = torch.concat([fin_prompt_seq, sample_token[None,...]], dim=-1)
        
#         if debug:
#             print(f"Accepted continuations: {tokenizer.decode(fin_prompt_seq[0,n_orig:], skip_special_tokens=True)}")

#         n += 1

#     return fin_prompt_seq
    
def speculative_sampling(target_model, draft_model, initial_prompt_seq, max_new_tokens, tokenizer, lookahead=16, temperature=1.0, debug=True):
    '''
    Implementation of Algorithm 2 of the paper - Accelerating Large Language Model Decoding 
    with Speculative Sampling (https://arxiv.org/abs/2302.01318)
    '''
    assert initial_prompt_seq.shape[0] == 1, 'Batch size should be 1'

    n = initial_prompt_seq.shape[-1]
    fin_prompt_seq = initial_prompt_seq.detach().clone()
    sum_nr = 0
    sum = 0

    while n < max_new_tokens:
        n_orig = n
        N = fin_prompt_seq.shape[-1]
        draft_outputs, draft_logits = sample_from_draft_model(draft_model, fin_prompt_seq, new_tokens=lookahead, temperature=temperature)
        draft_outputs2, draft_logits2 = sample_from_draft_model(draft_model, fin_prompt_seq, new_tokens=lookahead, temperature=temperature)

        # print(draft_outputs, draft_outputs2)
        
        if debug:
            print(f"Possible continuations: {tokenizer.decode(draft_outputs[0,n_orig:], skip_special_tokens=True)}")

        draft_outputs_list = [draft_outputs, draft_outputs2]
        draft_logits_list = [draft_logits, draft_logits2]

        target_logits = target_model(draft_outputs).logits[:, -lookahead-1:, :]
        target_logits2 = target_model(draft_outputs2).logits[:, -lookahead-1:, :]

        target_logits_list = [target_logits, target_logits2]

        target_model_distribution = get_distribution(target_logits, temperature)
        target_model_distribution2 = get_distribution(target_logits2, temperature)
        draft_model_distribution = get_distribution(draft_logits, temperature)
        draft_model_distribution2 = get_distribution(draft_logits2, temperature)
        
        target_model_distribution_list = [target_model_distribution, target_model_distribution2]
        draft_model_distribution_list = [draft_model_distribution, draft_model_distribution2]

        accepted_flag = 1
        accepted_flag_list = [1, 1]

        nr = n_orig
        
        for t in range(lookahead):
            numerator_list = [target_model_distribution_list[i][:, t, draft_outputs_list[i][0, N+t]] for i in range(2)]
            denominator_list = [draft_model_distribution_list[i][:, t, draft_outputs_list[i][0, N+t]] for i in range(2)]
            ratio_list = [numerator_list[i] / denominator_list[i] for i in range(2)]
            uniform_distribution = torch.rand_like(numerator_list[0])
            ones_tensor = torch.ones_like(numerator_list[0])

            code_token = None
            for i in range(2):
                if accepted_flag_list[i] == 1:
                    token = draft_outputs_list[i][0, N+t]
                    print(t, i, code_token, token)
                    if code_token != None and code_token !=token:
                        accepted_flag_list[i] = 0
                        continue
                    elif code_token!= None and code_token ==token:
                        # print(code_token, token)
                        continue
                    else:
                        print(ratio_list[i], uniform_distribution)
                        if (uniform_distribution < torch.min(ones_tensor, ratio_list[i])).any():
                            nr += 1
                            code_token = token
                        else:
                            accepted_flag_list[i] = 0
                            for j in range(2):
                                if accepted_flag_list[j] == 1:
                                    print('------------------------------')
                                    new_dist = target_model_distribution_list[j][:, t, draft_outputs_list[j][0, N+t]] - draft_model_distribution_list[i][:, t, draft_outputs_list[i][0, N+t]]
                                    new_dist = torch.max(torch.zeros_like(new_dist), new_dist)
                                    print(new_dist)
                                    print(target_model_distribution_list[j][:, t, draft_outputs_list[j][0, N+t]])
                                    print(draft_model_distribution_list[i][:, t, draft_outputs_list[j][0, N+t]])
                                    print(draft_outputs_list[i][0, N+t])
                                    print('------------------------------')
                                    ratio_list[j] = new_dist / draft_model_distribution_list[j][:, t, draft_outputs_list[j][0, N+t]]
                                    new_dist = None

            if all(x == 0 for x in accepted_flag_list):
                break
        nr += 1
        sum_nr += nr - n_orig
        print('pre',nr - n_orig)
                

        draft_outputs, draft_logits = sample_from_draft_model(draft_model, fin_prompt_seq, new_tokens=lookahead, temperature=temperature)
        
        if debug:
            print(f"Possible continuations: {tokenizer.decode(draft_outputs[0,n_orig:], skip_special_tokens=True)}")

        target_logits = target_model(draft_outputs).logits[:, -lookahead-1:, :]

        target_model_distribution = get_distribution(target_logits, temperature)
        draft_model_distribution = get_distribution(draft_logits, temperature)

        accepted_flag = 1
        
        for t in range(lookahead):
            numerator = target_model_distribution[:, t, draft_outputs[0, N+t]]
            denominator = draft_model_distribution[:, t, draft_outputs[0, N+t]]
            ratio = (numerator / denominator)
            uniform_distribution = torch.rand_like(numerator)
            ones_tensor = torch.ones_like(numerator)

            # Rejection Sampling
            ## Acceptance
            if (uniform_distribution < torch.min(ones_tensor, ratio)).any():
                fin_prompt_seq = torch.concat([fin_prompt_seq, draft_outputs[:, N+t].unsqueeze(dim=-1)], dim=-1)
                n += 1

            ## Rejection
            else:
                new_dist = (target_model_distribution[:, t, :] - draft_model_distribution[:, t, :])
                new_dist = torch.max(torch.zeros_like(new_dist), new_dist)
                new_dist = new_dist / new_dist.sum(dim=-1, keepdim=True)
                token_id = torch.multinomial(new_dist, num_samples=1)[0]
                fin_prompt_seq = torch.concat([fin_prompt_seq, token_id[None,...]], dim=-1)
                accepted_flag = 0
                break

        if accepted_flag == 1:
            sample_token = sample(target_logits[:, -1, :], temperature=temperature)
            fin_prompt_seq = torch.concat([fin_prompt_seq, sample_token[None,...]], dim=-1)
        
        if debug:
            print(f"Accepted continuations: {tokenizer.decode(fin_prompt_seq[0,n_orig:], skip_special_tokens=True)}")

        n += 1

        sum += n-n_orig
        
        print('next',n-n_orig)
        print(sum_nr, sum)

    return fin_prompt_seq
