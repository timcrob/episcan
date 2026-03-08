import argparse
import os
import torch
import whisper
from tvdb_v4_official import TVDB
import tmdbsimple as tmdb
import re
import subprocess
import tempfile
import pysubs2
import warnings
import numpy as np
from scipy.optimize import linear_sum_assignment
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import time
import json
from pathlib import Path
from platformdirs import user_cache_dir
try:
    from subliminal import download_best_subtitles, save_subtitles
    from subliminal.video import Episode
    from subliminal.core import provider_manager
    from subliminal.cache import region as subliminal_region
    subliminal_region.configure('dogpile.cache.memory')
    SUBLIMINAL_AVAILABLE = True
except ImportError:
    SUBLIMINAL_AVAILABLE = False

# ANSI color codes
class Colors:
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'

def colorize_similarity(similarity):
    """Return colored similarity score based on confidence level"""
    if similarity >= 0.7:
        return f"{Colors.GREEN}{similarity:.3f}{Colors.END}"
    elif similarity >= 0.5:
        return f"{Colors.YELLOW}{similarity:.3f}{Colors.END}"
    elif similarity >= 0.3:
        return f"{Colors.CYAN}{similarity:.3f}{Colors.END}"
    else:
        return f"{Colors.RED}{similarity:.3f}{Colors.END}"

def is_correctly_named(video_path, episode_info):
    """Check if video file is already correctly named"""
    current_name = os.path.splitext(os.path.basename(video_path))[0]
    target_name = re.sub(r'[<>:"/\|?*]', '', episode_info)
    
    # Normalize both names for comparison (remove extra spaces, case insensitive)
    current_normalized = re.sub(r'\s+', ' ', current_name.lower().strip())
    target_normalized = re.sub(r'\s+', ' ', target_name.lower().strip())
    
    # Check exact match first
    if current_normalized == target_normalized:
        return True
    
    # Also check if current name ends with the target (for cases like "Show Name - S01E01 - Episode" vs "S01E01 - Episode")
    if current_normalized.endswith(target_normalized):
        return True
    
    # Check if they match after removing common show name prefixes
    # Look for patterns like "Show Name - " at the beginning
    show_prefix_pattern = r'^[^-]+ - '
    current_without_show = re.sub(show_prefix_pattern, '', current_normalized).strip()
    target_without_show = re.sub(show_prefix_pattern, '', target_normalized).strip()
    
    return current_without_show == target_without_show

# Suppress the FP16 warning from Whisper
warnings.filterwarnings("ignore", message="FP16 is not supported on CPU; using FP32 instead")

def get_cache_dir():
    """Get or create cache directory for subtitles using OS-appropriate location"""
    cache_dir = Path(user_cache_dir("episcan", "episcan"))
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir

def get_cache_key(show_name, season, episode_number):
    """Generate cache key for episode subtitles"""
    # Normalize show name for filename safety
    safe_show = re.sub(r'[<>:"/\|?*]', '', show_name).strip()
    return f"{safe_show}_S{season:02d}E{episode_number:02d}"

def save_subtitle_to_cache(show_name, season, episode_number, subtitle_content, subtitle_text, provider_name):
    """Save subtitle content and metadata to cache"""
    try:
        cache_dir = get_cache_dir()
        cache_key = get_cache_key(show_name, season, episode_number)
        
        # Save raw subtitle content
        subtitle_file = cache_dir / f"{cache_key}.srt"
        with open(subtitle_file, 'w', encoding='utf-8') as f:
            f.write(subtitle_content)
        
        # Save metadata
        metadata = {
            'show_name': show_name,
            'season': season,
            'episode': episode_number,
            'provider': provider_name,
            'cached_at': time.time(),
            'text_length': len(subtitle_text)
        }
        
        metadata_file = cache_dir / f"{cache_key}.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
        
        return True
    except Exception as e:
        return False

def load_subtitle_from_cache(show_name, season, episode_number, verbose=False):
    """Load subtitle from cache if available"""
    try:
        cache_dir = get_cache_dir()
        cache_key = get_cache_key(show_name, season, episode_number)
        
        subtitle_file = cache_dir / f"{cache_key}.srt"
        metadata_file = cache_dir / f"{cache_key}.json"
        
        if subtitle_file.exists() and metadata_file.exists():
            # Load metadata
            with open(metadata_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            # Load subtitle content
            with open(subtitle_file, 'r', encoding='utf-8') as f:
                subtitle_content = f.read()
            
            # Parse subtitle text
            subtitle_text = parse_subtitle_content(subtitle_content)
            
            if subtitle_text and verbose:
                provider = metadata.get('provider', 'unknown')
                print(f"    ✓ Found cached subtitles ({len(subtitle_text)} chars) from {provider}")
            
            return subtitle_content, subtitle_text, metadata
        
    except Exception as e:
        if verbose:
            print(f"    Cache load error: {e}")
    
    return None, None, None

def clear_subtitle_cache(verbose=False):
    """Clear all cached subtitles"""
    try:
        cache_dir = get_cache_dir()
        count = 0
        for file in cache_dir.glob("*"):
            if file.suffix in ['.srt', '.json']:
                file.unlink()
                count += 1
        
        if verbose:
            print(f"Cleared {count} cached files")
        return count
    except Exception as e:
        if verbose:
            print(f"Error clearing cache: {e}")
        return 0

def main():
    args = get_args()
    
    # Handle cache management
    if args.clear_cache:
        print("Clearing subtitle cache...")
        cleared = clear_subtitle_cache(args.verbose)
        print(f"Cleared {cleared} cached files")
        if not args.video_dir or args.video_dir == ".":
            return  # If only clearing cache, exit
    
    # Auto-enable subtitle comparison unless descriptions are explicitly requested
    if not args.use_descriptions:
        args.use_subtitles_comparison = True
        if args.verbose and not args.subtitles_dir and not args.use_subliminal:
            print("Defaulting to subtitle comparison using subliminal (use --use-descriptions to compare against episode descriptions)")
    
    # Adjust transcription defaults based on comparison method
    if args.use_descriptions and args.max_duration == 180:
        # For description comparison, use full episode by default
        args.max_duration = None
        if args.verbose:
            print("Using full episode transcription for description comparison")
    elif args.use_subtitles_comparison and args.max_duration == 180:
        # Keep the shorter default for subtitle comparison
        if args.verbose:
            print(f"Using {args.max_duration}s transcription starting at {args.start_offset}s for subtitle comparison")
    
    # Determine which API to use based on available keys and preferences
    tmdb_key = args.tmdb_api_key or os.getenv('TMDB_API_KEY')
    tvdb_key = args.tvdb_api_key or os.getenv('TVDB_API_KEY')
    
    use_tmdb = False
    api_key = None
    
    if args.force_tvdb and tvdb_key:
        # Force TVDB if explicitly requested and available
        use_tmdb = False
        api_key = tvdb_key
        print("Using TVDB API (forced)")
    elif tmdb_key:
        # Default to TMDB if available
        use_tmdb = True
        api_key = tmdb_key
        print("Using TMDB API")
    elif tvdb_key:
        # Fall back to TVDB
        use_tmdb = False
        api_key = tvdb_key
        print("Using TVDB API")
    else:
        print("Error: Either TMDB or TVDB API key is required.")
        print("Set TMDB_API_KEY or TVDB_API_KEY environment variable, or use --tmdb-api-key or --tvdb-api-key arguments.")
        return
        
    # Load SBERT model for text similarity
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.verbose:
        print(f"Loading sentence transformer model ({args.sbert_model}) on {device}...")
    model = SentenceTransformer(args.sbert_model, device=device)
    
    # Whisper model will be loaded on-demand when needed
    
    # Get video files
    video_paths = get_video_files(args.video_dir)
    print(f"Found {len(video_paths)} video files")
    
    # Parse show/season from directory structure
    show_info = parse_plex_structure(video_paths[0] if video_paths else args.video_dir)
    print(f"Detected: {show_info['show']} Season {show_info['season']}")
    
    # Get episode data from selected API
    if use_tmdb:
        episodes_data, proper_show_name = get_tmdb_episodes(show_info, api_key, args.verbose)
    else:
        episodes_data, proper_show_name = get_tvdb_episodes(show_info, api_key, args.verbose)
        
    if proper_show_name:
        show_info['show'] = proper_show_name  # Use the proper name from API
    
    # Get subtitle comparison data (default behavior)
    if args.use_subtitles_comparison:
        if args.subtitles_dir:
            # Use local subtitle files
            print(f"Loading subtitles from directory: {args.subtitles_dir}")
            episodes_data = load_local_episode_subtitles(show_info, episodes_data, args.subtitles_dir, args.verbose)
        elif SUBLIMINAL_AVAILABLE:
            # Use subliminal to download subtitles
            print(f"Fetching episode subtitles using subliminal...")
            episodes_data = get_subliminal_episode_subtitles(show_info, episodes_data, args.verbose, args.no_cache)
        else:
            # No subtitle source available
            print("Warning: Subliminal library not available and no local subtitles provided. Install subliminal with: uv add subliminal")
            print("Falling back to episode descriptions.")
            args.use_subtitles_comparison = False
    
    print(f"Found {len(episodes_data)} episodes for {show_info['show']} Season {show_info['season']}")
    
    # Check subtitle coverage if using subtitle comparison
    if args.use_subtitles_comparison:
        episodes_with_subtitles = sum(1 for ep in episodes_data if 'subtitle_text' in ep and ep['subtitle_text'])
        total_episodes = len(episodes_data)
        
        if episodes_with_subtitles < total_episodes:
            missing_count = total_episodes - episodes_with_subtitles
            print(f"\n{Colors.YELLOW}⚠ Warning: Only {episodes_with_subtitles}/{total_episodes} episodes have subtitles ({missing_count} missing){Colors.END}")
            
            # Retry with specified number of attempts
            max_retries = args.subtitle_retries
            if max_retries > 0:
                print(f"{Colors.CYAN}Retrying subtitle download with {max_retries} attempts...{Colors.END}")
                episodes_data = retry_missing_subtitles(show_info, episodes_data, max_retries, args.verbose, args.no_cache)
                
                # Check coverage again after retries
                final_episodes_with_subtitles = sum(1 for ep in episodes_data if 'subtitle_text' in ep and ep['subtitle_text'])
                final_missing = total_episodes - final_episodes_with_subtitles
                
                if final_missing == 0:
                    print(f"{Colors.GREEN}✓ All episodes now have subtitles after retries!{Colors.END}")
                elif final_missing < missing_count:
                    print(f"{Colors.YELLOW}✓ Found subtitles for {missing_count - final_missing} more episodes. {final_missing} still missing.{Colors.END}")
                else:
                    print(f"{Colors.YELLOW}⚠ {final_missing} episodes still missing subtitles after {max_retries} retries.{Colors.END}")
                
                # Update missing count for final action decision
                missing_count = final_missing
            
            # Handle remaining missing subtitles based on failure action
            if missing_count > 0:
                if args.on_subtitle_failure == 'exit':
                    print(f"{Colors.RED}Exiting due to missing subtitles. Use --on-subtitle-failure=continue to proceed anyway.{Colors.END}")
                    return
                elif args.on_subtitle_failure == 'prompt':
                    response = input(f"\n{Colors.YELLOW}Continue with {missing_count} missing subtitles? (y/n): {Colors.END}").strip().lower()
                    if response not in ['y', 'yes']:
                        print(f"{Colors.YELLOW}Operation cancelled{Colors.END}")
                        return
                    else:
                        print(f"{Colors.GREEN}Continuing with available subtitles...{Colors.END}")
                elif args.on_subtitle_failure == 'continue':
                    print(f"{Colors.YELLOW}Continuing with {missing_count} missing subtitles...{Colors.END}")
        else:
            print(f"{Colors.GREEN}✓ All episodes have subtitles{Colors.END}")
    
    # Process each video file and collect all transcripts
    video_transcripts = {}
    print(f"Processing {len(video_paths)} video files...")
    
    start_time = time.time()
    
    for i, video_path in enumerate(video_paths, 1):
        video_start = time.time()
        
        if args.verbose:
            print(f"\n{Colors.BLUE}Processing ({i}/{len(video_paths)}): {Colors.BOLD}{os.path.basename(video_path)}{Colors.END}")
        else:
            # Calculate ETA
            if i > 1:
                elapsed = time.time() - start_time
                avg_time_per_video = elapsed / (i - 1)
                remaining_videos = len(video_paths) - i + 1
                eta_seconds = avg_time_per_video * remaining_videos
                eta_minutes = int(eta_seconds // 60)
                eta_secs = int(eta_seconds % 60)
                eta_str = f" (ETA: {eta_minutes:02d}:{eta_secs:02d})"
            else:
                eta_str = ""
            
            print(f"  {i}/{len(video_paths)}: {os.path.basename(video_path)}{eta_str}", end="")
        
        # Try to extract subtitles first if requested, otherwise use Whisper
        transcript = None
        if args.try_subtitles:
            # Pass episodes_data and model for potential similarity matching
            transcript = extract_subtitles(video_path, args.verbose, episodes_data, model, args)
        
        if not transcript:
            if args.verbose:
                if args.try_subtitles:
                    print("  No subtitles found, transcribing with Whisper...")
                else:
                    print("  Transcribing with Whisper...")
            transcript = transcribe_audio(video_path, args.whisper_model, args.max_duration, args.start_offset, args.verbose, device)
        else:
            if args.verbose:
                print("  Using extracted subtitles")
        
        if transcript:
            video_transcripts[video_path] = transcript
            if args.verbose:
                print("  Transcript extracted successfully")
            else:
                video_time = time.time() - video_start
                print(f" ✓ ({video_time:.1f}s)")
        else:
            if args.verbose:
                print("  Failed to get transcript")
            else:
                video_time = time.time() - video_start
                print(f" ✗ ({video_time:.1f}s)")
    
    # Find best matches ensuring no episode is matched to multiple videos
    print(f"\nCalculating optimal matches...")
    matches = find_unique_episode_matches(video_transcripts, episodes_data, model, args, args.verbose)
    
    # Print results for each video
    for video_path in video_paths:
        if video_path in matches:
            match = matches[video_path]
            is_correct = is_correctly_named(video_path, match['episode'])
            
            if is_correct:
                status_color = Colors.GREEN
                status_icon = "✓"
                status_text = "(correctly named)"
            else:
                status_color = Colors.RED
                status_icon = "→"
                status_text = "(needs renaming)"
            
            similarity_colored = colorize_similarity(match['similarity'])
            print(f"  {status_color}{status_icon} Best match: {match['episode']} {status_text}{Colors.END}")
            print(f"    Similarity: {similarity_colored}")
        else:
            print(f"  {Colors.RED}✗ No good match found for {os.path.basename(video_path)}{Colors.END}")
    
    # Print final results
    print(f"\n{Colors.BOLD}{Colors.UNDERLINE}=== FINAL MATCHES ==={Colors.END}")
    for video_path, match in matches.items():
        is_correct = is_correctly_named(video_path, match['episode'])
        
        if is_correct:
            filename_color = Colors.GREEN
            status_icon = "✓"
        else:
            filename_color = Colors.RED
            status_icon = "→"
        
        similarity_colored = colorize_similarity(match['similarity'])
        
        print(f"{filename_color}{status_icon} {os.path.basename(video_path)} -> {match['episode']}{Colors.END}")
        print(f"  Similarity: {similarity_colored}")
        print()
    
    # Handle file renaming based on user preference
    if matches and args.rename != 'none':
        rename_files(matches, args.rename, show_info)

def get_video_files(video_dir):
    """Get all video files from directory"""
    video_extensions = {'.mp4', '.m4v', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm'}
    video_paths = []
    
    for name in os.listdir(video_dir):
        file_path = os.path.join(video_dir, name)
        if os.path.isfile(file_path) and any(name.lower().endswith(ext) for ext in video_extensions):
            video_paths.append(file_path)
    
    return video_paths

def parse_plex_structure(video_path):
    """Parse Plex-style directory structure to extract show and season info"""
    # Normalize path separators
    path = os.path.normpath(video_path)
    parts = path.split(os.sep)
    
    show_name = None
    season_num = None
    
    # Look for show name and season in path
    for i, part in enumerate(parts):
        # Look for "Season X" pattern
        season_match = re.search(r'Season\s+(\d+)', part, re.IGNORECASE)
        if season_match:
            season_num = int(season_match.group(1))
            # Show name is usually the parent directory
            if i > 0:
                show_name = parts[i-1]
            break
    
    # If no season found, try to get show from parent directories
    if not show_name and len(parts) >= 2:
        show_name = parts[-2]  # Parent directory of the file
        season_num = 1  # Default to season 1
    
    return {
        'show': show_name or 'Unknown Show',
        'season': season_num or 1
    }

def get_tvdb_episodes(show_info, api_key, verbose=False):
    """Get episode data (descriptions) from TVDB API"""
    tvdb = TVDB(api_key)
    
    try:
        # Search for the show
        search_results = tvdb.search(show_info['show'])
        if not search_results:
            print(f"No results found for '{show_info['show']}'")
            return [], None
        
        series_id = search_results[0]['tvdb_id']
        proper_show_name = search_results[0]['name']
        if verbose:
            print(f"Found series: {proper_show_name} (ID: {series_id})")
        
        # Get all episodes for the series
        episodes_response = tvdb.get_series_episodes(series_id)
        
        # Handle different response formats
        if isinstance(episodes_response, dict) and 'episodes' in episodes_response:
            all_episodes = episodes_response['episodes']
        elif isinstance(episodes_response, dict) and 'data' in episodes_response:
            all_episodes = episodes_response['data']
        else:
            all_episodes = episodes_response
        
        # Filter episodes for the specific season and extract relevant data
        episodes_data = []
        for ep in all_episodes:
            season_num = ep.get('seasonNumber') or ep.get('season')
            if season_num == show_info['season']:
                episode_data = {
                    'number': ep.get('number', 0),
                    'name': ep.get('name', 'Unknown'),
                    'overview': ep.get('overview', ''),
                    'episode_id': f"S{show_info['season']:02d}E{ep.get('number', 0):02d} - {ep.get('name', 'Unknown')}"
                }
                episodes_data.append(episode_data)
        
        return episodes_data, proper_show_name
        
    except Exception as e:
        print(f"TVDB API error: {e}")
        return [], None

def get_tmdb_episodes(show_info, api_key, verbose=False):
    """Get episode data (descriptions) from TMDB API"""
    tmdb.API_KEY = api_key
    
    try:
        # Search for the show
        search = tmdb.Search()
        response = search.tv(query=show_info['show'])
        
        if not response['results']:
            print(f"No results found for '{show_info['show']}'")
            return [], None
        
        # Get the first result
        show = response['results'][0]
        show_id = show['id']
        proper_show_name = show['name']
        if verbose:
            print(f"Found series: {proper_show_name} (ID: {show_id})")
        
        # Get season details to get episodes
        tv_seasons = tmdb.TV_Seasons(show_id, show_info['season'])
        season_details = tv_seasons.info()
        
        # Extract episode data
        episodes_data = []
        for ep in season_details.get('episodes', []):
            episode_data = {
                'number': ep.get('episode_number', 0),
                'name': ep.get('name', 'Unknown'),
                'overview': ep.get('overview', ''),
                'episode_id': f"S{show_info['season']:02d}E{ep.get('episode_number', 0):02d} - {ep.get('name', 'Unknown')}"
            }
            episodes_data.append(episode_data)
        
        return episodes_data, proper_show_name
        
    except Exception as e:
        print(f"TMDB API error: {e}")
        return [], None

def get_subliminal_episode_subtitles(show_info, episodes_data, verbose=False, no_cache=False):
    """Get episode subtitles using subliminal library with caching support"""
    import tempfile
    from pathlib import Path
    
    subtitles_data = []
    cache_hits = 0
    downloads = 0
    
    if verbose:
        print(f"  Attempting to get subtitles for {len(episodes_data)} episodes (checking cache first)...")
    
    for episode in episodes_data:
        if verbose:
            print(f"  Processing Episode {episode['number']}...")
        
        # Check cache first (unless disabled)
        cached_content, cached_text, metadata = None, None, None
        if not no_cache:
            cached_content, cached_text, metadata = load_subtitle_from_cache(
                show_info['show'], show_info['season'], episode['number'], verbose
            )
        
        if cached_content and cached_text:
            # Use cached subtitles
            episode_with_subtitles = episode.copy()
            episode_with_subtitles['subtitle_text'] = cached_text
            episode_with_subtitles['subtitle_content'] = cached_content
            subtitles_data.append(episode_with_subtitles)
            cache_hits += 1
            continue
        
        # Not in cache, download with subliminal
        if verbose:
            print(f"    No cache found, downloading...")
        
        # Create temporary directory for video simulation  
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            try:
                # Create a fake video file for subliminal to work with
                episode_name = f"{show_info['show']}.S{show_info['season']:02d}E{episode['number']:02d}.mkv"
                fake_video_path = temp_path / episode_name
                fake_video_path.touch()
                
                # Create Episode object for subliminal
                video = Episode(
                    name=str(fake_video_path),
                    series=show_info['show'],
                    season=show_info['season'],
                    episodes=[episode['number']]
                )
                
                # Use OpenSubtitles VIP provider directly
                subtitles = {}
                best_subtitle = None
                try:
                    from subliminal.providers.opensubtitlescom import OpenSubtitlesComVipProvider
                    from babelfish import Language
                    if verbose:
                        print(f"    Trying opensubtitlescomvip...")
                    provider = OpenSubtitlesComVipProvider(
                        username=os.getenv('OPENSUBTITLES_USERNAME', ''),
                        password=os.getenv('OPENSUBTITLES_PASSWORD', ''),
                        apikey=os.getenv('OPENSUBTITLES_API_KEY', '')
                    )
                    provider.user_agent = 'episcan v1.0'
                    with provider:
                        results = provider.query(
                            {Language('eng')},
                            query=show_info['show'],
                            season=show_info['season'],
                            episode=episode['number']
                        )
                        if results:
                            best_subtitle = max(results, key=lambda s: getattr(s, 'score', 0))
                            provider.download_subtitle(best_subtitle)
                            subtitles = {video: [best_subtitle]}
                            if verbose:
                                print(f"    Found and downloaded subtitle")
                        else:
                            if verbose:
                                print(f"    ✗ No subtitles found")
                except Exception as provider_error:
                    if verbose:
                        print(f"    Provider failed: {provider_error}")
                
                if subtitles and video in subtitles and subtitles[video]:
                    try:
                        subtitle_text = best_subtitle.content.decode('utf-8') if best_subtitle.content else ''
                    except UnicodeDecodeError:
                        try:
                            subtitle_text = best_subtitle.content.decode('latin-1') if best_subtitle.content else ''
                        except:
                            subtitle_text = ''
                    
                    if subtitle_text:
                        clean_text = parse_subtitle_content(subtitle_text)
                        
                        if clean_text:
                            if not no_cache:
                                save_subtitle_to_cache(
                                    show_info['show'], 
                                    show_info['season'], 
                                    episode['number'],
                                    subtitle_text,
                                    clean_text,
                                    best_subtitle.provider_name
                                )
                            
                            episode_with_subtitles = episode.copy()
                            episode_with_subtitles['subtitle_text'] = clean_text
                            episode_with_subtitles['subtitle_content'] = subtitle_text
                            subtitles_data.append(episode_with_subtitles)
                            downloads += 1
                            
                            if verbose:
                                print(f"    ✓ Downloaded and cached subtitles ({len(clean_text)} chars) from {best_subtitle.provider_name}")
                        else:
                            subtitles_data.append(episode)
                            if verbose:
                                print(f"    ✗ Failed to extract text from subtitle")
                    else:
                        subtitles_data.append(episode)
                        if verbose:
                            print(f"    ✗ Empty subtitle content")
                else:
                    subtitles_data.append(episode)
                    if verbose:
                        print(f"    ✗ No subtitles found from any provider")
                
                # Clean up the fake file
                fake_video_path.unlink()
                
            except Exception as e:
                # Error, use description
                subtitles_data.append(episode)
                if verbose:
                    print(f"    ✗ Error: {e}")
            
            # Rate limiting - be respectful to providers
            time.sleep(1.0)
    
    # Summary
    subtitle_count = sum(1 for ep in subtitles_data if 'subtitle_text' in ep)
    if verbose:
        print(f"  Results: {cache_hits} from cache, {downloads} downloaded, {subtitle_count}/{len(episodes_data)} total with subtitles")
    
    return subtitles_data

def retry_missing_subtitles(show_info, episodes_data, max_retries, verbose=False, no_cache=False):
    """Retry downloading subtitles for episodes that don't have them, with exponential backoff"""
    import tempfile
    from pathlib import Path
    
    missing_episodes = [ep for ep in episodes_data if 'subtitle_text' not in ep or not ep['subtitle_text']]
    if not missing_episodes:
        return episodes_data
    
    if verbose:
        print(f"\n{Colors.YELLOW}Retrying subtitle download for {len(missing_episodes)} episodes...{Colors.END}")
    
    for retry_attempt in range(max_retries):
        if not missing_episodes:
            break
            
        backoff_delay = 2 * (2 ** retry_attempt)  # Exponential backoff: 2s, 4s, 8s, 16s...
        
        if verbose:
            print(f"\n  Retry attempt {retry_attempt + 1}/{max_retries} (delay: {backoff_delay}s)...")
        
        if retry_attempt > 0:
            print(f"    Waiting {backoff_delay} seconds...")
            time.sleep(backoff_delay)
        
        newly_found = []
        still_missing = []
        
        for episode in missing_episodes:
            if verbose:
                print(f"    Retrying Episode {episode['number']}...")
            
            # Try again with same logic as original download
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                
                try:
                    # Create enhanced fake video file
                    clean_episode_name = re.sub(r'[<>:"/\\|?*]', '', episode.get('name', 'Unknown')).strip()
                    episode_filename = f"{show_info['show']}.S{show_info['season']:02d}E{episode['number']:02d}.{clean_episode_name}.mkv"
                    fake_video_path = temp_path / episode_filename
                    fake_video_path.touch()
                    
                    # Create Episode object
                    video = Episode(
                        name=str(fake_video_path),
                        series=show_info['show'],
                        season=show_info['season'],
                        episodes=[episode['number']],
                        title=episode.get('name', 'Unknown'),
                        year=episode.get('year'),
                        imdb_id=episode.get('imdb_id'),
                        size=0,
                    )
                    
                    # Try providers with longer delay
                    try:
                        all_providers = list(provider_manager.names())
                        reliable_providers = ['opensubtitles', 'podnapisi', 'addic7ed']
                        reliable_providers = [p for p in reliable_providers if p in all_providers]
                    except Exception:
                        all_providers = None
                        reliable_providers = ['opensubtitles', 'podnapisi']
                    
                    # Only try reliable providers for retries
                    subtitles = download_best_subtitles(
                        [video], 
                        languages={'en'},
                        providers=reliable_providers,
                        provider_configs={
                            'opensubtitles': {
                                'username': os.getenv('OPENSUBTITLES_USERNAME', ''), 
                                'password': os.getenv('OPENSUBTITLES_PASSWORD', '')
                            },
                            'addic7ed': {
                                'username': os.getenv('ADDIC7ED_USERNAME', ''),
                                'password': os.getenv('ADDIC7ED_PASSWORD', '')
                            }
                        }
                    )
                    
                    if subtitles and video in subtitles and subtitles[video]:
                        best_subtitle = max(subtitles[video], key=lambda s: getattr(s, 'score', 0))
                        
                        try:
                            subtitle_text = best_subtitle.content.decode('utf-8') if best_subtitle.content else ''
                        except UnicodeDecodeError:
                            try:
                                subtitle_text = best_subtitle.content.decode('latin-1') if best_subtitle.content else ''
                            except:
                                subtitle_text = ''
                        
                        if subtitle_text:
                            clean_text = parse_subtitle_content(subtitle_text)
                            
                            if clean_text:
                                # Save to cache
                                if not no_cache:
                                    save_subtitle_to_cache(
                                        show_info['show'], 
                                        show_info['season'], 
                                        episode['number'],
                                        subtitle_text,
                                        clean_text,
                                        best_subtitle.provider_name
                                    )
                                
                                # Update episode data
                                episode['subtitle_text'] = clean_text
                                episode['subtitle_content'] = subtitle_text
                                newly_found.append(episode)
                                
                                if verbose:
                                    print(f"      ✓ Found subtitles ({len(clean_text)} chars) from {best_subtitle.provider_name}")
                                continue
                    
                    # Still no subtitles found
                    still_missing.append(episode)
                    if verbose:
                        print(f"      ✗ Still no subtitles found")
                        
                except Exception as e:
                    still_missing.append(episode)
                    if verbose:
                        print(f"      ✗ Error: {e}")
                
                # Rate limiting between episodes
                time.sleep(1.0)
        
        missing_episodes = still_missing
        
        if newly_found:
            if verbose:
                print(f"    Found subtitles for {len(newly_found)} more episodes")
        
        if not missing_episodes:
            if verbose:
                print(f"    ✓ All episodes now have subtitles!")
            break
    
    if missing_episodes and verbose:
        print(f"    ✗ {len(missing_episodes)} episodes still missing subtitles after {max_retries} retries")
    
    return episodes_data

def parse_subtitle_content(content):
    """Parse subtitle content and extract clean text using pysubs2"""
    try:
        # Use pysubs2 to parse subtitle content (supports SRT, VTT, ASS, etc.)
        subs = pysubs2.SSAFile.from_string(content)
        text_lines = [event.plaintext for event in subs if event.plaintext.strip()]
        return ' '.join(text_lines)
    except Exception as e:
        # Fallback to manual parsing if pysubs2 fails
        lines = content.split('\n')
        text_lines = []
        
        for line in lines:
            line = line.strip()
            # Skip empty lines, numbers, timestamps, and format markers
            if (line and 
                not line.isdigit() and 
                not re.match(r'\d{2}:\d{2}:\d{2}', line) and
                not line.startswith('WEBVTT') and
                not line.startswith('NOTE') and
                not re.match(r'\d+$', line)):
                # Clean HTML tags and formatting
                clean_line = re.sub(r'<[^>]+>', '', line)
                clean_line = re.sub(r'\{[^}]+\}', '', clean_line)
                clean_line = re.sub(r'\\N', ' ', clean_line)  # ASS format line breaks
                clean_line = clean_line.strip()
                if clean_line:
                    text_lines.append(clean_line)
        
        return ' '.join(text_lines)

def load_local_episode_subtitles(show_info, episodes_data, subtitles_dir, verbose=False):
    """Load episode subtitles from local directory"""
    if not os.path.isdir(subtitles_dir):
        print(f"Subtitles directory not found: {subtitles_dir}")
        return episodes_data
    
    subtitle_extensions = {'.srt', '.vtt', '.ass', '.ssa'}
    subtitles_data = []
    
    # Get all subtitle files in directory
    subtitle_files = []
    for filename in os.listdir(subtitles_dir):
        if any(filename.lower().endswith(ext) for ext in subtitle_extensions):
            subtitle_files.append(os.path.join(subtitles_dir, filename))
    
    if verbose:
        print(f"  Found {len(subtitle_files)} subtitle files in directory")
    
    for episode in episodes_data:
        episode_with_subtitles = episode.copy()
        
        # Try to find matching subtitle file
        matched_file = find_matching_subtitle_file(
            episode, show_info, subtitle_files, verbose
        )
        
        if matched_file:
            subtitle_text = parse_subtitle_file(matched_file, verbose)
            if subtitle_text:
                episode_with_subtitles['subtitle_text'] = subtitle_text
                if verbose:
                    print(f"  ✓ {episode['episode_id']}: {os.path.basename(matched_file)} ({len(subtitle_text)} chars)")
            else:
                if verbose:
                    print(f"  ✗ {episode['episode_id']}: Failed to parse {os.path.basename(matched_file)}")
        else:
            if verbose:
                print(f"  ✗ {episode['episode_id']}: No matching subtitle file found")
        
        subtitles_data.append(episode_with_subtitles)
    
    return subtitles_data

def find_matching_subtitle_file(episode, show_info, subtitle_files, verbose=False):
    """Find subtitle file that matches the episode"""
    season_str = f"S{show_info['season']:02d}"
    episode_str = f"E{episode['number']:02d}"
    episode_patterns = [
        f"{season_str}{episode_str}",  # S01E05
        f"{show_info['season']}x{episode['number']:02d}",  # 1x05
        f"Season {show_info['season']} Episode {episode['number']}",  # Season 1 Episode 5
        f"s{show_info['season']:02d}e{episode['number']:02d}",  # s01e05
        episode['name'].lower().replace(' ', '.'),  # episode.name
    ]
    
    # Score each subtitle file
    best_match = None
    best_score = 0
    
    for subtitle_file in subtitle_files:
        filename = os.path.basename(subtitle_file).lower()
        score = 0
        
        # Check for episode patterns
        for pattern in episode_patterns:
            if pattern.lower() in filename:
                score += 10
                break
        
        # Check for show name
        if show_info['show'].lower().replace(' ', '.') in filename.replace(' ', '.'):
            score += 5
        
        # Prefer files with exact season/episode match
        if f"{season_str.lower()}{episode_str.lower()}" in filename:
            score += 20
        
        if score > best_score:
            best_score = score
            best_match = subtitle_file
    
    return best_match if best_score > 0 else None

def parse_subtitle_file(subtitle_path, verbose=False):
    """Parse subtitle file and extract text content using pysubs2"""
    try:
        # Use pysubs2 to load and parse any supported subtitle format
        subs = pysubs2.load(subtitle_path)
        text_lines = [event.plaintext for event in subs if event.plaintext.strip()]
        return ' '.join(text_lines)
    except Exception as e:
        if verbose:
            print(f"    Error parsing {subtitle_path}: {e}")
        return None

def extract_subtitles(video_path, verbose=False, episodes_data=None, model=None, args=None):
    """Extract subtitle text from video file, finding the track with best similarity to episode subtitles"""
    try:
        # Try to extract subtitle tracks using ffmpeg
        result = subprocess.run([
            'ffprobe', '-v', 'quiet', '-print_format', 'json', 
            '-show_streams', video_path
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            import json
            data = json.loads(result.stdout)
            
            # Look for subtitle streams
            subtitle_streams = [s for s in data.get('streams', []) if s.get('codec_type') == 'subtitle']
            
            if subtitle_streams:
                # Find the best matching subtitle track
                # If we have episode data, try to find best similarity match across all episodes
                best_episode_match = None
                if episodes_data and model and args and args.use_subtitles_comparison:
                    # Find which episode has subtitles that best match any of our video subtitle tracks
                    best_episode_match = find_best_episode_subtitle_match(subtitle_streams, episodes_data, model, video_path, verbose)
                
                # Get the best subtitle stream (either similarity-based or fallback to English)
                episode_subtitle_text = best_episode_match['subtitle_text'] if best_episode_match and 'subtitle_text' in best_episode_match else None
                best_stream = find_best_subtitle_stream(subtitle_streams, verbose, episode_subtitle_text, model)
                
                if best_stream is not None:
                    stream_index = best_stream['index']
                    
                    # Extract selected subtitle stream to SRT format
                    with tempfile.NamedTemporaryFile(suffix='.srt', delete=False) as tmp:
                        extract_result = subprocess.run([
                            'ffmpeg', '-i', video_path, '-map', f'0:s:{stream_index}', 
                            '-c:s', 'srt', tmp.name, '-y'
                        ], capture_output=True)
                        
                        # Read the SRT file if extraction was successful
                        if extract_result.returncode == 0:
                            try:
                                subs = pysubs2.load(tmp.name)
                                text = ' '.join([event.plaintext for event in subs])
                                os.unlink(tmp.name)
                                if verbose:
                                    lang = best_stream.get('tags', {}).get('language', 'unknown')
                                    title = best_stream.get('tags', {}).get('title', '')
                                    print(f"  Using subtitle track {stream_index}: {lang} {title}".strip())
                                return text
                            except Exception as e:
                                if verbose:
                                    print(f"  Failed to parse SRT file: {e}")
                                os.unlink(tmp.name)
                        else:
                            os.unlink(tmp.name)
                            if verbose:
                                print(f"  Failed to extract subtitle track {stream_index}")
        
        return None
    except Exception as e:
        if verbose:
            print(f"  Subtitle extraction failed: {e}")
        return None

def find_best_subtitle_stream(subtitle_streams, verbose=False, episode_subtitle_text=None, model=None):
    """Find the best subtitle stream by comparing with episode subtitles, fallback to English preference"""
    if not subtitle_streams:
        return None
    
    # If we have episode subtitle text and model, try similarity-based matching first
    if episode_subtitle_text and model:
        if verbose:
            print("  Comparing subtitle tracks with episode subtitles...")
        
        best_similarity_score = 0
        best_similarity_stream = None
        similarity_results = []
        
        for i, stream in enumerate(subtitle_streams):
            # Extract this subtitle track's text
            try:
                stream_text = extract_subtitle_track_text(stream, video_path, i, verbose)
                if stream_text and len(stream_text.strip()) > 50:  # Minimum content check
                    # Calculate similarity with episode subtitles
                    from sklearn.metrics.pairwise import cosine_similarity
                    
                    # Encode both texts
                    stream_embedding = model.encode([stream_text.lower().strip()])
                    episode_embedding = model.encode([episode_subtitle_text.lower().strip()])
                    
                    # Calculate similarity
                    similarity = cosine_similarity(stream_embedding, episode_embedding)[0][0]
                    similarity_results.append((similarity, i, stream, stream_text))
                    
                    if verbose:
                        tags = stream.get('tags', {})
                        lang = tags.get('language', 'unknown')
                        print(f"    Track {i} ({lang}): similarity {similarity:.3f}")
                    
                    if similarity > best_similarity_score:
                        best_similarity_score = similarity
                        best_similarity_stream = stream
                        best_similarity_stream['stream_index'] = i
                else:
                    if verbose:
                        print(f"    Track {i}: failed to extract or insufficient content")
            except Exception as e:
                if verbose:
                    print(f"    Track {i}: error extracting - {e}")
        
        # If we found a good similarity match (threshold: 0.3), use it
        if best_similarity_score > 0.3:
            if verbose:
                print(f"  Selected track {best_similarity_stream['stream_index']} based on similarity: {best_similarity_score:.3f}")
            return best_similarity_stream
        elif verbose:
            print(f"  No good similarity match found (best: {best_similarity_score:.3f}), falling back to English preference")
    
    # Fallback to English preference scoring
    if verbose:
        print("  Using English preference scoring...")
    
    # Score each subtitle stream
    scored_streams = []
    
    for i, stream in enumerate(subtitle_streams):
        tags = stream.get('tags', {})
        language = tags.get('language', '').lower()
        title = tags.get('title', '').lower()
        codec_name = stream.get('codec_name', '').lower()
        
        score = 0
        
        # English language preference (highest priority for fallback)
        if language in ['eng', 'en', 'english']:
            score += 100
        elif language == '':
            # No language specified - could be English, give some points
            score += 50
        
        # Title scoring - look for English indicators
        english_indicators = ['english', 'eng', 'en']
        for indicator in english_indicators:
            if indicator in title:
                score += 30
                break
                break
        
        # Prefer non-forced subtitles
        if 'forced' not in title:
            score += 20
        
        # Prefer SDH (Subtitles for Deaf and Hard of hearing) for completeness
        if 'sdh' in title:
            score += 10
        
        # Codec preference - text-based codecs are better
        if codec_name in ['srt', 'ass', 'ssa', 'subrip']:
            score += 15
        elif codec_name in ['webvtt', 'ttml']:
            score += 10
        # PGS, DVB subtitles are image-based and harder to extract
        elif codec_name in ['pgs', 'dvb_subtitle']:
            score -= 20
        
        # Prefer earlier streams if all else equal (usually main language)
        score += (100 - i)  # Earlier streams get slightly higher score
        
        scored_streams.append((score, i, stream))
        
        if verbose:
            lang_desc = language or 'unknown'
            title_desc = title or 'no title'
            print(f"    Track {i}: {lang_desc} '{title_desc}' ({codec_name}) - score: {score}")
    
    # Sort by score (highest first) and return the best stream
    scored_streams.sort(reverse=True, key=lambda x: x[0])
    
    if scored_streams:
        best_score, best_index, best_stream = scored_streams[0]
        
        # Add the stream index to the stream object for easier reference
        best_stream['stream_index'] = best_index
        
        if verbose:
            print(f"  Selected best subtitle track: {best_index} (score: {best_score})")
        
        return best_stream
    
    return None

def extract_subtitle_track_text(stream, video_path, stream_index, verbose=False):
    """Extract text content from a specific subtitle track"""
    try:
        with tempfile.NamedTemporaryFile(suffix='.srt', delete=False) as tmp:
            # Extract specific subtitle stream to SRT format
            extract_result = subprocess.run([
                'ffmpeg', '-i', video_path, '-map', f'0:s:{stream_index}', 
                '-c:s', 'srt', tmp.name, '-y'
            ], capture_output=True, stderr=subprocess.DEVNULL)
            
            if extract_result.returncode == 0:
                try:
                    subs = pysubs2.load(tmp.name)
                    text = ' '.join([event.plaintext for event in subs if event.plaintext.strip()])
                    os.unlink(tmp.name)
                    return text
                except Exception as e:
                    if verbose:
                        print(f"    Failed to parse subtitle track {stream_index}: {e}")
                    os.unlink(tmp.name)
            else:
                os.unlink(tmp.name)
                if verbose:
                    print(f"    Failed to extract subtitle track {stream_index}")
    except Exception as e:
        if verbose:
            print(f"    Error extracting track {stream_index}: {e}")
    
    return None

def find_best_episode_subtitle_match(subtitle_streams, episodes_data, model, video_path, verbose=False):
    """Find which episode's subtitles best match any of the video's subtitle tracks"""
    try:
        if verbose:
            print("  Comparing video subtitle tracks with all episode subtitles...")
        
        best_match_score = 0
        best_episode = None
        
        # First, extract all video subtitle tracks
        video_subtitles = []
        for i, stream in enumerate(subtitle_streams):
            stream_text = extract_subtitle_track_text(stream, video_path, i, verbose)
            if stream_text and len(stream_text.strip()) > 50:
                video_subtitles.append((i, stream_text))
        
        if not video_subtitles:
            if verbose:
                print("    No extractable subtitle tracks found")
            return None
        
        # Compare each video subtitle track against each episode's subtitles
        for episode in episodes_data:
            if 'subtitle_text' not in episode or not episode['subtitle_text']:
                continue
                
            episode_text = episode['subtitle_text'].lower().strip()
            
            for track_idx, video_text in video_subtitles:
                try:
                    from sklearn.metrics.pairwise import cosine_similarity
                    
                    # Encode both texts
                    video_embedding = model.encode([video_text.lower().strip()])
                    episode_embedding = model.encode([episode_text])
                    
                    # Calculate similarity
                    similarity = cosine_similarity(video_embedding, episode_embedding)[0][0]
                    
                    if verbose:
                        print(f"    Track {track_idx} vs {episode['episode_id']}: {similarity:.3f}")
                    
                    if similarity > best_match_score:
                        best_match_score = similarity
                        best_episode = episode
                
                except Exception as e:
                    if verbose:
                        print(f"    Error comparing track {track_idx} with {episode['episode_id']}: {e}")
        
        if best_episode and best_match_score > 0.4:  # Good match threshold
            if verbose:
                print(f"  Best episode match: {best_episode['episode_id']} (similarity: {best_match_score:.3f})")
            return best_episode
        elif verbose:
            print(f"  No good episode match found (best: {best_match_score:.3f})")
            
    except Exception as e:
        if verbose:
            print(f"  Error in episode matching: {e}")
    
    return None

def extract_subtitle_time_segment(subtitle_content_or_path, start_offset_seconds, duration_seconds, verbose=False):
    """Extract subtitle text for a specific time segment"""
    try:
        # If we have subtitle content directly (from subliminal)
        if isinstance(subtitle_content_or_path, str) and ('\n' in subtitle_content_or_path or 'WEBVTT' in subtitle_content_or_path):
            # Parse subtitle content directly
            subs = pysubs2.SSAFile.from_string(subtitle_content_or_path)
        elif subtitle_content_or_path and os.path.exists(subtitle_content_or_path):
            # Load from file path
            subs = pysubs2.load(subtitle_content_or_path)
        else:
            # No valid subtitle source
            return None
        
        start_ms = start_offset_seconds * 1000
        end_ms = (start_offset_seconds + duration_seconds) * 1000
        
        # Extract events within the time range
        segment_events = []
        for event in subs:
            # Check if event overlaps with our time segment
            if event.start < end_ms and event.end > start_ms:
                segment_events.append(event.plaintext)
        
        if segment_events:
            segment_text = ' '.join(segment_events)
            if verbose:
                print(f"    Extracted {len(segment_text)} chars from {start_offset_seconds}s-{start_offset_seconds + duration_seconds}s")
            return segment_text
            
    except Exception as e:
        if verbose:
            print(f"    Failed to extract time segment: {e}")
    
    return None

def transcribe_audio(video_path, whisper_model_name, max_duration=None, start_offset=0, verbose=False, device="cpu"):
    """Transcribe audio using Whisper (optionally limit duration and start offset)"""
    try:
        # Load Whisper model on-demand
        if verbose:
            print(f"  Loading Whisper model ({whisper_model_name}) on {device}...")
        whisper_model = whisper.load_model(whisper_model_name, device=device)
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            # Build ffmpeg command
            ffmpeg_cmd = ['ffmpeg', '-i', video_path]
            
            # Add start offset if specified
            if start_offset > 0:
                ffmpeg_cmd.extend(['-ss', str(start_offset)])
            
            # Add duration limit if specified
            if max_duration is not None:
                ffmpeg_cmd.extend(['-t', str(max_duration)])
                if verbose:
                    if start_offset > 0:
                        print(f"  Transcribing {max_duration} seconds starting at {start_offset}s...")
                    else:
                        print(f"  Transcribing first {max_duration} seconds...")
            else:
                if verbose:
                    if start_offset > 0:
                        print(f"  Transcribing from {start_offset}s to end...")
                    else:
                        print("  Transcribing entire episode...")
            
            # Add audio extraction parameters
            ffmpeg_cmd.extend([
                '-vn', '-acodec', 'pcm_s16le', '-ar', '16000', 
                tmp.name, '-y'
            ])
            
            # Extract audio with progress indicator
            if verbose:
                print("  Extracting audio...", end="", flush=True)
            
            start_time = time.time()
            subprocess.run(ffmpeg_cmd, capture_output=True)
            
            if verbose:
                extraction_time = time.time() - start_time
                print(f" done ({extraction_time:.1f}s)")
                print("  Running Whisper transcription...", end="", flush=True)
            
            # Transcribe with Whisper
            transcribe_start = time.time()
            result = whisper_model.transcribe(tmp.name)
            os.unlink(tmp.name)
            
            if verbose:
                transcribe_time = time.time() - transcribe_start
                total_time = time.time() - start_time
                print(f" done ({transcribe_time:.1f}s total: {total_time:.1f}s)")
            
            return result['text']
    except Exception as e:
        if verbose:
            print(f"  Whisper transcription failed: {e}")
        return None

def find_unique_episode_matches(video_transcripts, episodes_data, model, args, verbose=False):
    """Find best matching episodes using optimal assignment algorithm with SBERT"""
    if not video_transcripts or not episodes_data:
        return {}
    
    video_paths = list(video_transcripts.keys())
    
    # Create similarity matrix: videos x episodes
    similarity_matrix = np.zeros((len(video_paths), len(episodes_data)))
    
    if verbose:
        print(f"  Building similarity matrix ({len(video_paths)} videos × {len(episodes_data)} episodes)...")
    
    # Prepare all texts for batch encoding
    transcripts = [video_transcripts[video_path].lower().strip() for video_path in video_paths]
    
    # Use subtitles if available (default), otherwise use descriptions
    episode_texts = []
    comparison_method = "subtitles" if args.use_subtitles_comparison else "descriptions"
    
    for episode in episodes_data:
        if not args.use_descriptions and 'subtitle_text' in episode and episode['subtitle_text']:
            # Use actual episode subtitles for comparison (default)
            episode_subtitle_text = episode['subtitle_text'].lower().strip()
            
            # If we're using partial transcription, extract corresponding time segment from subtitles
            if args.max_duration and args.max_duration > 0:
                # Extract subtitle segment matching the transcription time range
                segment_text = extract_subtitle_time_segment(
                    episode.get('subtitle_content'),  # Use stored raw content
                    args.start_offset, 
                    args.max_duration,
                    verbose
                )
                if segment_text:
                    episode_texts.append(segment_text.lower().strip())
                    if verbose and len(episode_texts) == 1:  # Only print once
                        print(f"  Extracting {args.max_duration}s subtitle segments starting at {args.start_offset}s for fair comparison")
                else:
                    # Fallback to full subtitles if time extraction fails
                    episode_texts.append(episode_subtitle_text)
            else:
                # Use full episode subtitles (for full transcription or max_duration=0)
                episode_texts.append(episode_subtitle_text)
        else:
            # Use episode descriptions
            episode_texts.append(f"{episode['name']} {episode['overview']}".lower().strip())
    
    if verbose:
        print(f"  Comparison method: {comparison_method}")
    
    # Encode all texts in batches for efficiency
    if verbose:
        print("  Encoding transcripts...", end="", flush=True)
    
    start_time = time.time()
    transcript_embeddings = model.encode(transcripts)
    
    if verbose:
        encode_time = time.time() - start_time
        print(f" done ({encode_time:.1f}s)")
        print("  Encoding episode descriptions...", end="", flush=True)
    
    episode_start = time.time()
    episode_embeddings = model.encode(episode_texts)
    
    if verbose:
        episode_time = time.time() - episode_start
        total_encoding_time = time.time() - start_time
        print(f" done ({episode_time:.1f}s, total: {total_encoding_time:.1f}s)")
    
    # Calculate similarity matrix using cosine similarity
    if verbose:
        print("  Calculating similarity matrix...", end="", flush=True)
    
    sim_start = time.time()
    similarity_matrix = cosine_similarity(transcript_embeddings, episode_embeddings)
    
    if verbose:
        sim_time = time.time() - sim_start
        print(f" done ({sim_time:.1f}s)")
    
    # Show detailed scores in verbose mode
    if verbose:
        print("  Similarity scores:")
        for i, video_path in enumerate(video_paths):
            print(f"    {os.path.basename(video_path)}:")
            for j, episode in enumerate(episodes_data):
                score = similarity_matrix[i, j]
                print(f"      {episode['episode_id']}: {score:.3f}")
        print()
    
    # Find optimal assignment using Hungarian algorithm
    # Note: linear_sum_assignment minimizes, so we use negative similarities
    row_indices, col_indices = linear_sum_assignment(-similarity_matrix)
    
    if verbose:
        print(f"  Optimal assignment found!")
    
    # Create final matches from optimal assignment
    final_matches = {}
    total_similarity = 0
    
    for video_idx, episode_idx in zip(row_indices, col_indices):
        similarity = similarity_matrix[video_idx, episode_idx]
        
        # Only include matches above threshold
        if similarity > 0.1:
            video_path = video_paths[video_idx]
            episode = episodes_data[episode_idx]
            
            final_matches[video_path] = {
                'episode': episode['episode_id'],
                'similarity': similarity
            }
            total_similarity += similarity
            
            if verbose:
                print(f"  Assigned: {os.path.basename(video_path)} → {episode['episode_id']} (similarity: {similarity:.3f})")
    
    if verbose:
        print(f"  Final matches: {len(final_matches)} (total similarity: {total_similarity:.3f})")
    return final_matches

def find_best_episode_match(transcript, episodes_data, model):
    """Find best matching episode using SBERT similarity"""
    if not transcript or not episodes_data:
        return None
    
    # Clean and prepare transcript
    transcript = transcript.lower().strip()
    
    # Prepare episode texts
    episode_texts = [f"{episode['name']} {episode['overview']}".lower().strip() for episode in episodes_data]
    
    # Encode transcript and episodes
    transcript_embedding = model.encode([transcript])
    episode_embeddings = model.encode(episode_texts)
    
    # Calculate similarities
    similarities = cosine_similarity(transcript_embedding, episode_embeddings)[0]
    
    # Find best match
    best_idx = np.argmax(similarities)
    best_similarity = similarities[best_idx]
    
    if best_similarity > 0.1:  # Minimum threshold
        return {
            'episode': episodes_data[best_idx]['episode_id'],
            'similarity': best_similarity
        }
    
    return None

def rename_files(matches, rename_mode, show_info):
    """Rename video files based on episode matches using Plex-compatible naming with conflict resolution"""
    import uuid
    
    print(f"\n{Colors.BOLD}{Colors.UNDERLINE}=== FILE RENAMING ==={Colors.END}")
    
    # Build rename mapping
    rename_map = {}  # old_path -> new_path
    filename_map = {}  # old_path -> new_filename (for display)
    
    for video_path, match in matches.items():
        # Get file extension
        _, ext = os.path.splitext(video_path)
        
        # Parse episode info from episode_id (e.g., "S01E05 - Nicknames")
        episode_info = match['episode']
        
        # Create Plex-compatible filename: "Show Name - S01E01 - Episode Name.ext"
        show_name = show_info['show']
        
        # Clean show name and episode info for filename (remove invalid characters)
        clean_show = re.sub(r'[<>:"/\|?*]', '', show_name)
        clean_episode = re.sub(r'[<>:"/\|?*]', '', episode_info)
        
        # Format: "Show Name - S01E01 - Episode Name"
        new_filename = f"{clean_show} - {clean_episode}{ext}"
        
        # Get directory and create new path
        video_dir = os.path.dirname(video_path)
        new_path = os.path.join(video_dir, new_filename)
        
        # Skip if filename would be the same
        if video_path == new_path:
            continue
            
        rename_map[video_path] = new_path
        filename_map[video_path] = new_filename
    
    if not rename_map:
        print(f"  {Colors.GREEN}✓ No files need renaming{Colors.END}")
        return
    
    # Show planned renames
    print(f"  {Colors.BOLD}Planned renames:{Colors.END}")
    conflicts = 0
    for old_path, new_path in rename_map.items():
        new_filename = filename_map[old_path]
        if os.path.exists(new_path):
            conflicts += 1
            print(f"    {Colors.RED}{os.path.basename(old_path)}{Colors.END} {Colors.WHITE}→{Colors.END} {Colors.YELLOW}{new_filename} (conflict){Colors.END}")
        else:
            print(f"    {Colors.RED}{os.path.basename(old_path)}{Colors.END} {Colors.WHITE}→{Colors.END} {Colors.GREEN}{new_filename}{Colors.END}")
    
    if conflicts:
        print(f"\n  {Colors.YELLOW}⚠ {conflicts} conflict(s) - existing files will be preserved with UUID suffix{Colors.END}")
    
    # Handle based on mode
    if rename_mode == 'prompt':
        response = input(f"\n  {Colors.YELLOW}Proceed with renaming? (y/n): {Colors.END}").strip().lower()
        if response not in ['y', 'yes']:
            print(f"  {Colors.YELLOW}Renaming cancelled{Colors.END}")
            return
    
    # Execute renames
    print(f"\n  {Colors.BOLD}Renaming files...{Colors.END}")
    
    # Track what we've moved for the final rename phase
    moved_sources = {}  # original_source_path -> current_source_path
    preserved_files = []
    
    try:
        # Step 1: Move any conflicting target files out of the way
        for old_path, new_path in rename_map.items():
            if os.path.exists(new_path):
                # Create unique name for the conflicting file
                base_name, ext = os.path.splitext(os.path.basename(new_path))
                conflict_filename = f"{base_name}_{str(uuid.uuid4())[:8]}{ext}"
                conflict_path = os.path.join(os.path.dirname(new_path), conflict_filename)
                
                # Check if this target file is also a source file in our rename map
                # If so, we need to track where it's been moved to
                for source_path, target_path in rename_map.items():
                    if source_path == new_path:
                        moved_sources[source_path] = conflict_path
                        break
                
                os.rename(new_path, conflict_path)
                preserved_files.append((os.path.basename(new_path), conflict_filename))
                print(f"    {Colors.CYAN}📁 Preserved: {os.path.basename(new_path)} -> {conflict_filename}{Colors.END}")
        
        # Step 2: Do all the renames
        success_count = 0
        for old_path, new_path in rename_map.items():
            try:
                # Check if this source file was already moved in step 1
                current_source = moved_sources.get(old_path, old_path)
                
                os.rename(current_source, new_path)
                print(f"    {Colors.GREEN}✓ {os.path.basename(current_source)} -> {filename_map[old_path]}{Colors.END}")
                success_count += 1
                
            except Exception as e:
                print(f"    {Colors.RED}✗ Failed: {os.path.basename(old_path)} - {e}{Colors.END}")
                
    except Exception as e:
        print(f"    {Colors.RED}✗ Critical error: {e}{Colors.END}")
    
    # Summary
    summary_parts = [f"{Colors.GREEN}{success_count}{Colors.END}{Colors.BOLD}/{len(rename_map)} files renamed"]
    if preserved_files:
        summary_parts.append(f"{Colors.CYAN}{len(preserved_files)} files preserved{Colors.END}")
    
    print(f"\n  {Colors.BOLD}{', '.join(summary_parts)}{Colors.END}")

def get_args():
    parser = argparse.ArgumentParser(
        prog='episcan',
        description='''Match TV episodes using audio transcription and episode subtitles (default) or descriptions.
        
Subtitle sources (automatic detection):
  1. Local subtitle files (--subtitles-dir)  
  2. Subliminal library (default) - Multiple providers, no API key needed

Transcription defaults:
  - Subtitle comparison: 3 minutes starting at 1 minute (skips intros)
  - Description comparison: Full episode transcription

Subtitle Coverage:
  Script retries missing subtitles 5 times by default, then exits if any still missing
  Use --subtitle-retries and --on-subtitle-failure to customize behavior

Examples:
  %(prog)s /path/to/videos  # Uses subliminal, 3min excerpt, 5 retries then exit
  %(prog)s /path/to/videos --use-descriptions  # Full episode vs descriptions
  %(prog)s /path/to/videos --subtitle-retries=10  # 10 retry attempts
  %(prog)s /path/to/videos --on-subtitle-failure=continue  # Continue if missing
  %(prog)s /path/to/videos --subtitle-retries=0  # No retries, exit immediately
        ''',
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("video_dir", nargs='?', default=".", help="Directory containing video files")
    parser.add_argument('--tvdb-api-key', help="TVDB API key (can also use TVDB_API_KEY environment variable)")
    parser.add_argument('--tmdb-api-key', help="TMDB API key (can also use TMDB_API_KEY environment variable)")
    parser.add_argument('--subtitles-dir', help="Directory containing subtitle files for episodes")
    parser.add_argument('--use-descriptions', action='store_true', help="Use episode descriptions for comparison instead of subtitles (default: use subtitles)")
    parser.add_argument('--use-subliminal', action='store_true', help="Force use of subliminal library (enabled by default when no local subtitles specified)")
    parser.add_argument('--use-subtitles-comparison', action='store_true', help="Use episode subtitles for comparison (default behavior, kept for compatibility)")
    parser.add_argument('--force-tvdb', action='store_true', help="Force use of TVDB even if TMDB key is available")
    parser.add_argument('--whisper-model', default='base', choices=['tiny', 'base', 'small', 'medium', 'large'], 
                       help="Whisper model size (default: base)")
    parser.add_argument('--sbert-model', default='sentence-transformers/all-mpnet-base-v2', 
                       choices=[
                           'sentence-transformers/all-mpnet-base-v2',
                           'sentence-transformers/all-MiniLM-L6-v2',
                           'sentence-transformers/all-MiniLM-L12-v2',
                           'sentence-transformers/paraphrase-mpnet-base-v2',
                           'sentence-transformers/multi-qa-mpnet-base-dot-v1'
                       ],
                       help="SBERT model for text similarity (default: all-mpnet-base-v2)")
    parser.add_argument('--max-duration', type=int, default=180, help="Maximum duration in seconds to transcribe (default: 180 for subtitle comparison, use 0 for full episode)")
    parser.add_argument('--start-offset', type=int, default=60, help="Start transcription offset in seconds to skip openings (default: 60)")
    parser.add_argument('--rename', choices=['none', 'prompt', 'auto'], default='none', 
                       help="File renaming behavior: none (default), prompt (ask user), auto (rename automatically)")
    parser.add_argument('--verbose', action='store_true', help="Show detailed processing information")
    parser.add_argument('--try-subtitles', action='store_true', help="Try to extract subtitles first, fallback to Whisper if not found")
    parser.add_argument('--clear-cache', action='store_true', help="Clear subtitle cache before processing")
    parser.add_argument('--no-cache', action='store_true', help="Disable subtitle caching (always download fresh)")
    parser.add_argument('--subtitle-retries', type=int, default=5,
                       help="Number of retry attempts for missing subtitles (default: 5)")
    parser.add_argument('--on-subtitle-failure', choices=['exit', 'prompt', 'continue'], default='exit',
                       help="Action when subtitles still missing after retries: exit (default), prompt user, or continue anyway")
    return parser.parse_args()

if __name__ == "__main__":
    main()