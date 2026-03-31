import json
import mimetypes
import os
import re
import shlex
import subprocess
import tempfile
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup


class ToolManager:
    def __init__(self, safe_mode=True):
        self.safe_mode = safe_mode
        self.dangerous_commands = [
            'rm', 'rmdir', 'del', 'format', 'fdisk', 'mkfs',
            'shutdown', 'reboot', 'halt', 'poweroff',
            'sudo rm', 'sudo rmdir', 'sudo del',
            'chmod 777', 'chown root'
        ]
        self.allowed_file_extensions = [
            '.txt', '.md', '.json', '.yaml', '.yml', '.csv',
            '.log', '.config', '.conf', '.ini', '.py', '.js',
            '.html', '.css', '.xml'
        ]

    def execute_system_commands(self, command: str, safety_check: bool = True) -> Dict[str, Any]:
        try:
            if safety_check and self.safe_mode:
                safety_result = self._safety_check_command(command)
                if not safety_result['safe']:
                    return {
                        'success': False,
                        'output': '',
                        'error': f"Command blocked for safety: {safety_result['reason']}",
                        'command': command
                    }
            
            command_parts = shlex.split(command)
            
            result = subprocess.run(
                command_parts,
                capture_output=True,
                text=True,
                timeout=30,
                cwd=os.getcwd()
            )
            
            return {
                'success': result.returncode == 0,
                'output': result.stdout,
                'error': result.stderr,
                'return_code': result.returncode,
                'command': command
            }
            
        except subprocess.TimeoutExpired:
            return {
                'success': False,
                'output': '',
                'error': 'Command timed out after 30 seconds',
                'command': command
            }
        except Exception as e:
            return {
                'success': False,
                'output': '',
                'error': f'Error executing command: {str(e)}',
                'command': command
            }

    def file_operations(self, action: str, path: str, content: str = None) -> Dict[str, Any]:
        try:
            if self.safe_mode and not self._is_path_safe(path):
                return {
                    'success': False,
                    'error': f'Path not allowed in safe mode: {path}'
                }
            
            if action.lower() == 'read':
                return self._read_file(path)
            elif action.lower() == 'write':
                return self._write_file(path, content)
            elif action.lower() == 'create':
                return self._create_file_or_directory(path, content)
            elif action.lower() == 'delete':
                return self._delete_file_or_directory(path)
            elif action.lower() == 'list':
                return self._list_directory(path)
            elif action.lower() == 'exists':
                return self._check_exists(path)
            else:
                return {
                    'success': False,
                    'error': f'Unknown action: {action}. Available: read, write, create, delete, list, exists'
                }
                
        except Exception as e:
            return {
                'success': False,
                'error': f'File operation error: {str(e)}'
            }

    def web_scraping(self, url: str, extract_type: str = "text") -> Dict[str, Any]:
        try:
            if not self._is_url_safe(url):
                return {
                    'success': False,
                    'error': 'URL not allowed or potentially unsafe'
                }
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            if extract_type.lower() == "text":
                return self._extract_text_content(response, url)
            elif extract_type.lower() == "links":
                return self._extract_links(response, url)
            elif extract_type.lower() == "images":
                return self._extract_images(response, url)
            elif extract_type.lower() == "metadata":
                return self._extract_metadata(response, url)
            elif extract_type.lower() == "all":
                return self._extract_all_content(response, url)
            else:
                return {
                    'success': False,
                    'error': f'Unknown extract_type: {extract_type}. Available: text, links, images, metadata, all'
                }
                
        except requests.RequestException as e:
            return {
                'success': False,
                'error': f'Network error: {str(e)}'
            }
        except Exception as e:
            return {
                'success': False,
                'error': f'Web scraping error: {str(e)}'
            }

    def _safety_check_command(self, command: str) -> Dict[str, Any]:
        command_lower = command.lower().strip()
        
        for dangerous_cmd in self.dangerous_commands:
            if dangerous_cmd in command_lower:
                return {
                    'safe': False,
                    'reason': f'Contains dangerous command: {dangerous_cmd}'
                }
        
        if '>' in command and '/dev/' in command:
            return {
                'safe': False,
                'reason': 'Potential device file manipulation'
            }
        
        if any(char in command for char in ['|', ';', '&&', '||']) and len(command) > 50:
            return {
                'safe': False,
                'reason': 'Complex command chaining detected'
            }
        
        return {'safe': True, 'reason': 'Command appears safe'}

    def _is_path_safe(self, path: str) -> bool:
        if self.safe_mode:
            forbidden_paths = ['/etc', '/sys', '/proc', '/dev', '/boot', '/root']
            path_lower = path.lower()
            
            if any(forbidden in path_lower for forbidden in forbidden_paths):
                return False
            
            if path.startswith('/') and not path.startswith(os.path.expanduser('~')):
                return False
        
        return True

    def _is_url_safe(self, url: str) -> bool:
        try:
            parsed = urlparse(url)
            if parsed.scheme not in ['http', 'https']:
                return False
            
            forbidden_domains = ['localhost', '127.0.0.1', '0.0.0.0']
            if any(domain in parsed.netloc.lower() for domain in forbidden_domains):
                return False
            
            return True
        except:
            return False

    def _read_file(self, path: str) -> Dict[str, Any]:
        try:
            if not os.path.exists(path):
                return {'success': False, 'error': f'File not found: {path}'}
            
            if os.path.isdir(path):
                return {'success': False, 'error': f'Path is a directory: {path}'}
            
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            return {
                'success': True,
                'content': content,
                'path': path,
                'size': len(content)
            }
        except UnicodeDecodeError:
            return {'success': False, 'error': 'File contains non-text content'}
        except Exception as e:
            return {'success': False, 'error': str(e)}

    def _write_file(self, path: str, content: str) -> Dict[str, Any]:
        try:
            if content is None:
                return {'success': False, 'error': 'Content cannot be None for write operation'}
            
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            with open(path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            return {
                'success': True,
                'path': path,
                'size': len(content),
                'message': f'Successfully wrote {len(content)} characters to {path}'
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}

    def _create_file_or_directory(self, path: str, content: str = None) -> Dict[str, Any]:
        try:
            if content is None:
                os.makedirs(path, exist_ok=True)
                return {
                    'success': True,
                    'path': path,
                    'type': 'directory',
                    'message': f'Created directory: {path}'
                }
            else:
                return self._write_file(path, content)
        except Exception as e:
            return {'success': False, 'error': str(e)}

    def _delete_file_or_directory(self, path: str) -> Dict[str, Any]:
        try:
            if not os.path.exists(path):
                return {'success': False, 'error': f'Path not found: {path}'}
            
            if os.path.isfile(path):
                os.remove(path)
                return {
                    'success': True,
                    'path': path,
                    'type': 'file',
                    'message': f'Deleted file: {path}'
                }
            elif os.path.isdir(path):
                os.rmdir(path)
                return {
                    'success': True,
                    'path': path,
                    'type': 'directory',
                    'message': f'Deleted directory: {path}'
                }
        except Exception as e:
            return {'success': False, 'error': str(e)}

    def _list_directory(self, path: str) -> Dict[str, Any]:
        try:
            if not os.path.exists(path):
                return {'success': False, 'error': f'Directory not found: {path}'}
            
            if not os.path.isdir(path):
                return {'success': False, 'error': f'Path is not a directory: {path}'}
            
            items = []
            for item in os.listdir(path):
                item_path = os.path.join(path, item)
                item_info = {
                    'name': item,
                    'path': item_path,
                    'type': 'directory' if os.path.isdir(item_path) else 'file',
                    'size': os.path.getsize(item_path) if os.path.isfile(item_path) else None
                }
                items.append(item_info)
            
            return {
                'success': True,
                'path': path,
                'items': items,
                'count': len(items)
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}

    def _check_exists(self, path: str) -> Dict[str, Any]:
        try:
            exists = os.path.exists(path)
            result = {
                'success': True,
                'path': path,
                'exists': exists
            }
            
            if exists:
                result.update({
                    'is_file': os.path.isfile(path),
                    'is_directory': os.path.isdir(path),
                    'size': os.path.getsize(path) if os.path.isfile(path) else None
                })
            
            return result
        except Exception as e:
            return {'success': False, 'error': str(e)}

    def _extract_text_content(self, response, url: str) -> Dict[str, Any]:
        soup = BeautifulSoup(response.content, 'html.parser')
        
        for script in soup(["script", "style"]):
            script.decompose()
        
        text = soup.get_text()
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = ' '.join(chunk for chunk in chunks if chunk)
        
        return {
            'success': True,
            'url': url,
            'content_type': 'text',
            'text': text[:5000],
            'full_length': len(text),
            'title': soup.title.string if soup.title else 'No title'
        }

    def _extract_links(self, response, url: str) -> Dict[str, Any]:
        soup = BeautifulSoup(response.content, 'html.parser')
        
        links = []
        for link in soup.find_all('a', href=True):
            href = link['href']
            absolute_url = urljoin(url, href)
            links.append({
                'text': link.get_text(strip=True),
                'href': href,
                'absolute_url': absolute_url
            })
        
        return {
            'success': True,
            'url': url,
            'content_type': 'links',
            'links': links[:50],
            'total_links': len(links)
        }

    def _extract_images(self, response, url: str) -> Dict[str, Any]:
        soup = BeautifulSoup(response.content, 'html.parser')
        
        images = []
        for img in soup.find_all('img'):
            src = img.get('src')
            if src:
                absolute_url = urljoin(url, src)
                images.append({
                    'src': src,
                    'absolute_url': absolute_url,
                    'alt': img.get('alt', ''),
                    'title': img.get('title', '')
                })
        
        return {
            'success': True,
            'url': url,
            'content_type': 'images',
            'images': images[:20],
            'total_images': len(images)
        }

    def _extract_metadata(self, response, url: str) -> Dict[str, Any]:
        soup = BeautifulSoup(response.content, 'html.parser')
        
        metadata = {
            'title': soup.title.string if soup.title else 'No title',
            'description': '',
            'keywords': '',
            'author': '',
            'og_data': {},
            'twitter_data': {}
        }
        
        for meta in soup.find_all('meta'):
            name = meta.get('name', '').lower()
            property_attr = meta.get('property', '').lower()
            content = meta.get('content', '')
            
            if name == 'description':
                metadata['description'] = content
            elif name == 'keywords':
                metadata['keywords'] = content
            elif name == 'author':
                metadata['author'] = content
            elif property_attr.startswith('og:'):
                metadata['og_data'][property_attr] = content
            elif name.startswith('twitter:'):
                metadata['twitter_data'][name] = content
        
        return {
            'success': True,
            'url': url,
            'content_type': 'metadata',
            'metadata': metadata
        }

    def _extract_all_content(self, response, url: str) -> Dict[str, Any]:
        text_result = self._extract_text_content(response, url)
        links_result = self._extract_links(response, url)
        images_result = self._extract_images(response, url)
        metadata_result = self._extract_metadata(response, url)
        
        return {
            'success': True,
            'url': url,
            'content_type': 'all',
            'text': text_result.get('text', ''),
            'title': text_result.get('title', ''),
            'links': links_result.get('links', [])[:10],
            'images': images_result.get('images', [])[:5],
            'metadata': metadata_result.get('metadata', {}),
            'summary': {
                'text_length': text_result.get('full_length', 0),
                'total_links': links_result.get('total_links', 0),
                'total_images': images_result.get('total_images', 0)
            }
        }