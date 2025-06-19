import os
import sys
sys.path.append("/data1/hang/Stellantis/InternVL/internvl_chat")
from internvl.model import InternVLChatConfig, load_model_and_tokenizer
from internvl.train.dataset import build_transform, dynamic_preprocess
from PIL import Image
import argparse
import torch
import json
from tqdm import tqdm
import time


def inference_single_image(image_path, question, model, tokenizer, device, use_thumbnail=False, image_size=224, max_new_tokens=1000, max_num=6, num_beams=3, temperature=0.0):
    """Run inference on a single image"""
    try:
        # Load and preprocess image with dynamic sizing (matching training)
        image = Image.open(image_path).convert('RGB')
        
        # Use dynamic preprocessing like in training
        images = dynamic_preprocess(image, image_size=image_size,
                                  use_thumbnail=use_thumbnail,
                                  max_num=max_num)
        
        # Apply transforms
        transform = build_transform(is_train=False, input_size=image_size)
        pixel_values = [transform(img) for img in images]
        pixel_values = torch.stack(pixel_values)
        
        # Move to specified GPU device and convert to bfloat16
        pixel_values = pixel_values.to(torch.bfloat16).to(device)
        
        if question is None:
            question = '<image>'
        
        generation_config = dict(
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            temperature=temperature,
            do_sample=True if temperature > 0 else False,
            min_new_tokens=1,
        )
        
        with torch.no_grad():
            outputs = model.chat(
                tokenizer=tokenizer,
                pixel_values=pixel_values,
                question=question,
                generation_config=generation_config,
            )
            print('outputs', outputs)
            return outputs
    except Exception as e:
        print(f"Error processing image {image_path}: {str(e)}")
        return None


def process_validation_dataset(jsonl_path, model, tokenizer, args, output_file):
    """Process the entire validation dataset"""
    # Load all data from JSONL file
    print(f"Loading data from {jsonl_path}...")
    data = [json.loads(line) for line in open(jsonl_path).readlines()]
    print(f"Loaded {len(data)} instances")
    
    # Get model configuration
    image_size = model.config.force_image_size or model.config.vision_config.image_size
    use_thumbnail = model.config.use_thumbnail
    max_num = model.config.max_dynamic_patch
    
    print(f"Model config - Image size: {image_size}, Use thumbnail: {use_thumbnail}, Max patches: {max_num}")
    
    results = []
    
    # Check if output file exists to resume processing
    processed_ids = set()
    if os.path.exists(output_file):
        try:
            with open(output_file, 'r') as f:
                existing_results = json.load(f)
            processed_ids = {result['id'] for result in existing_results}
            results.extend(existing_results)
            print(f"Found existing results file with {len(existing_results)} instances. Resuming...")
        except:
            print("Could not load existing results, starting fresh...")
    
    # Process each instance
    save_counter = 0
    for i, instance in tqdm(enumerate(data), total=len(data), desc="Processing validation dataset"):
        instance_id = instance.get('id', i)
        
        # Skip if already processed
        if instance_id in processed_ids:
            continue
        
        image_path = instance['image']
        conversations = instance['conversations']
        
        # Extract human question and groundtruth answer
        human_question = None
        groundtruth = None
        
        for conv in conversations:
            if conv['from'] == 'human':
                human_question = conv['value']
            elif conv['from'] == 'gpt':
                groundtruth = conv['value']
        
        if human_question is None:
            print(f"Warning: No human question found for instance {instance_id}")
            continue
        
        # Check if image file exists
        if not os.path.exists(image_path):
            print(f"Warning: Image not found: {image_path}")
            continue
        
        # Run inference
        try:
            # Track inference time
            inference_start_time = time.time()
            
            prediction = inference_single_image(
                image_path=image_path,
                question=human_question,
                model=model,
                tokenizer=tokenizer,
                device=args.device,
                use_thumbnail=use_thumbnail,
                image_size=image_size,
                max_new_tokens=args.max_new_tokens,
                max_num=max_num,
                num_beams=args.num_beams,
                temperature=args.temperature,
            )
            
            inference_end_time = time.time()
            inference_runtime = inference_end_time - inference_start_time
            
            # Store result
            result = {
                'id': instance_id,
                'image_path': image_path,
                'question': human_question,
                'groundtruth': groundtruth,
                'prediction': prediction,
                'runtime_seconds': round(inference_runtime, 3),
                'metadata': {
                    'width': instance.get('width'),
                    'height': instance.get('height'),
                    'processed_at': time.strftime('%Y-%m-%d %H:%M:%S')
                }
            }
            results.append(result)
            save_counter += 1
            
            # Print progress for first few instances
            if i < 3 or (i + 1) % 10 == 0:
                print(f"\n--- Instance {i+1}/{len(data)} (ID: {instance_id}) ---")
                print(f"Image: {os.path.basename(image_path)}")
                print(f"Runtime: {inference_runtime:.3f}s")
                print(f"Prediction preview: {str(prediction)[:100]}...")
                if groundtruth:
                    print(f"Groundtruth preview: {str(groundtruth)[:100]}...")
            
            # Save results every 5 instances and show runtime stats
            if save_counter % 5 == 0:
                with open(output_file, 'w') as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)
                
                # Calculate runtime statistics
                recent_runtimes = [r['runtime_seconds'] for r in results[-5:]]
                avg_runtime = sum(recent_runtimes) / len(recent_runtimes)
                min_runtime = min(recent_runtimes)
                max_runtime = max(recent_runtimes)
                
                print(f"\n[SAVED] Progress saved after {save_counter} new instances (Total: {len(results)} instances)")
                print(f"Runtime stats (last 5): Avg={avg_runtime:.3f}s, Min={min_runtime:.3f}s, Max={max_runtime:.3f}s")
                
        except Exception as e:
            print(f"Error processing instance {instance_id}: {str(e)}")
            continue
    
    # Final save
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n[FINAL SAVE] All results saved to {output_file}")
    print(f"Processed {len(results)} instances total")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Batch inference on validation dataset")
    parser.add_argument('--jsonl-file', type=str, 
                        default='/data1/hang/Stellantis/InternVL/notebooks/vdn_chat_train_only_fieldtext.jsonl',
                        help='Path to validation JSONL file')
    parser.add_argument('--checkpoint', type=str, 
                        default="/data1/hang/Stellantis/InternVL/internvl_chat/work_dirs/internvl_chat_v3/vdn_fieldtext_internvl3_1b_dynamic_res_2nd_finetune_full",
                        help='Path to model checkpoint')
    parser.add_argument('--output-file', type=str,
                        default='train_results_vdn_fieldtext_internvl3_1b_dynamic_res_2nd_finetune_full_3beams.json',
                        help='Output file for results')
    parser.add_argument('--num-beams', type=int, default=3)
    parser.add_argument('--max-new-tokens', type=int, default=1000)
    parser.add_argument('--max-num', type=int, default=6)
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--auto', action='store_true', default=False, help='Use auto device mapping')
    parser.add_argument('--load-in-8bit', action='store_true', default=False, help='Load model in 8-bit mode')
    parser.add_argument('--load-in-4bit', action='store_true', default=False, help='Load model in 4-bit mode')
    parser.add_argument('--max-instances', type=int, default=None, help='Maximum number of instances to process (for testing)')
    parser.add_argument('--device', type=str, default='cuda', help='GPU device to use')
    
    args = parser.parse_args()
    
    print("="*80)
    print("BATCH INFERENCE ON VALIDATION DATASET")
    print("="*80)
    print(f"JSONL file: {args.jsonl_file}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Output file: {args.output_file}")
    print(f"Device: {args.device}")
    print(f"Max instances: {args.max_instances or 'All'}")
    print("="*80)
    
    # Validate device
    if args.device.startswith('cuda'):
        if not torch.cuda.is_available():
            print("Error: CUDA is not available but CUDA device was specified")
            return
        if ':' in args.device:
            device_id = int(args.device.split(':')[1])
            if device_id >= torch.cuda.device_count():
                print(f"Error: GPU device {device_id} not available. Available devices: 0-{torch.cuda.device_count()-1}")
                return
            print(f"Using GPU {device_id}: {torch.cuda.get_device_name(device_id)}")
        else:
            print(f"Using default CUDA device: {torch.cuda.get_device_name(0)}")
    else:
        print(f"Using device: {args.device}")
    
    # Check if JSONL file exists
    if not os.path.exists(args.jsonl_file):
        print(f"Error: JSONL file not found: {args.jsonl_file}")
        return
    
    # Load model and tokenizer
    print("Loading model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer(args)
    
    # Move model to specified device
    print(f"Moving model to device: {args.device}")
    model = model.to(args.device)
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters()) / 1e9
    image_size = model.config.force_image_size or model.config.vision_config.image_size
    use_thumbnail = model.config.use_thumbnail
    
    print(f'Model info:')
    print(f'  - Total parameters: {total_params:.1f}B')
    print(f'  - Image size: {image_size}')
    print(f'  - Use thumbnail: {use_thumbnail}')
    print(f'  - Template: {model.config.template}')
    print(f'  - Model device: {next(model.parameters()).device}')
    print("="*80)
    
    # Process validation dataset
    results = process_validation_dataset(
        jsonl_path=args.jsonl_file,
        model=model,
        tokenizer=tokenizer,
        args=args,
        output_file=args.output_file
    )
    
    # Limit results if specified
    if args.max_instances and len(results) > args.max_instances:
        results = results[:args.max_instances]
        with open(args.output_file, 'w') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"Limited results to {args.max_instances} instances")
    
    print("="*80)
    print("PROCESSING COMPLETED")
    print(f"Total instances processed: {len(results)}")
    print(f"Results saved to: {args.output_file}")
    
    # Calculate and display runtime statistics
    if results:
        runtimes = [r['runtime_seconds'] for r in results if 'runtime_seconds' in r]
        if runtimes:
            total_inference_time = sum(runtimes)
            avg_runtime = total_inference_time / len(runtimes)
            min_runtime = min(runtimes)
            max_runtime = max(runtimes)
            
            print("\n" + "="*80)
            print("RUNTIME STATISTICS")
            print("="*80)
            print(f"Total inference time: {total_inference_time:.2f}s ({total_inference_time/60:.1f} minutes)")
            print(f"Average per image: {avg_runtime:.3f}s")
            print(f"Fastest inference: {min_runtime:.3f}s")
            print(f"Slowest inference: {max_runtime:.3f}s")
            print(f"Throughput: {len(runtimes)/total_inference_time:.2f} images/second")
            print(f"Estimated time per 100 images: {avg_runtime*100:.1f}s ({avg_runtime*100/60:.1f} minutes)")
    
    print("="*80)


if __name__ == '__main__':
    main() 